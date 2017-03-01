import os
import time
import numpy as np
import keras.backend as K
import tensorflow as tf

import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

import matplotlib.pyplot as plt

from keras.layers.convolutional import Convolution2D, Deconvolution2D, UpSampling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Reshape, Flatten, Activation
from keras.layers import Input
from keras.models import Model
from keras import initializations
from keras.regularizers import l1, l2, l1l2

from tensorflow.examples.tutorials.mnist import input_data

# ----------------------------------------------------------------------------

default_opt = { 'lr' : 5e-5, 'c' : 1e-2, 'n_critic' : 5 }
n_chan = 1

class WDCGAN(object):
  """Wasserstein Deep Convolutional Generative Adversarial Network"""

  def __init__(self, n_dim, n_chan=1, opt_alg='rmsprop', opt_params=default_opt):
    # set up some default hyper-params
    n_lat = 100 # latent variables
    n_g_hid1 = 1024 # size of hidden layer in generator layer 1
    n_g_hid2 = 128  # size of hidden layer in generator layer 2
    n_out = n_dim * n_dim * n_chan # total dimensionality of output

    print opt_params
    self.n_critic = opt_params['n_critic']
    self.c        = opt_params['c']

    # create session
    self.sess = tf.Session()
    K.set_session(self.sess) # pass keras the session

    # create generator
    with tf.name_scope('generator'):
      Xk_g = Input(shape=(n_lat,))
      g = make_keras_dcgan_generator(Xk_g, n_lat)
      # g = make_tweaked_generator(Xk_g, n_lat)

    # create discriminator
    with tf.name_scope('discriminator'):
      Xk_d = Input(shape=(n_chan, n_dim, n_dim))
      d = make_keras_dcgan_discriminator(Xk_d)
      # d = make_tweaked_discriminator(Xk_d)

    # instantiate networks
    g_net = Model(input=Xk_g, output=g)
    d_net = Model(input=Xk_d, output=d)

    # save inputs
    X_g = tf.placeholder(tf.float32, shape=(None, n_lat), name='X_g')
    X_d = tf.placeholder(tf.float32, shape=(None, n_chan, n_dim, n_dim), name='X_d')
    self.inputs = X_g, X_d

    # # create generator
    # g = make_dcgan_generator(X_g, n_lat)
    # d_real = make_dcgan_discriminator(X_d)
    # d_fake = make_dcgan_discriminator(g, reuse=True)
    # self.P = g

    # get their weights
    # self.w_g = [w for l in g_net.layers for w in l.trainable_weights]
    # self.w_d = [w for l in d_net.layers for w in l.trainable_weights]

    self.w_g = [w for w in tf.global_variables() if 'generator' in w.name]
    self.w_d = [w for w in tf.global_variables() if 'discriminator' in w.name]

    # create predictions
    d_real = d_net(X_d)
    d_fake = d_net(g_net(X_g))
    self.P = g_net(X_g)

    # create losses
    self.loss_g = tf.reduce_mean(d_fake)
    self.loss_d = tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)

    # compute and store discriminator probabilities
    self.d_real = tf.reduce_mean(d_real)
    self.d_fake = tf.reduce_mean(d_fake)
    self.p_real = tf.reduce_mean(tf.sigmoid(d_real))
    self.p_fake = tf.reduce_mean(tf.sigmoid(d_fake))

    # create an optimizer
    lr = opt_params['lr']
    optimizer_g = tf.train.RMSPropOptimizer(lr)
    optimizer_d = tf.train.RMSPropOptimizer(lr)

    # get gradients
    gv_g = optimizer_g.compute_gradients(self.loss_g, self.w_g)
    gv_d = optimizer_d.compute_gradients(self.loss_d, self.w_d)

    # create training operation
    self.train_op_g = optimizer_g.apply_gradients(gv_g)
    self.train_op_d = optimizer_d.apply_gradients(gv_d)

    # clip the weights, so that they fall in [-c, c]
    self.clip_updates = [w.assign(tf.clip_by_value(w, -self.c, self.c)) for w in self.w_d]

  def fit(self, X_train, X_val, n_epoch=10, n_batch=128, logdir='dcgan-run'):
    # initialize log directory                  
    if tf.gfile.Exists(logdir): tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)

    mnist = input_data.read_data_sets('data/mnist')

    # # create a saver
    # checkpoint_root = os.path.join(logdir, 'model.ckpt')
    # saver = tf.train.Saver()

    # # summarization
    # summary = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)

    # init model
    init = tf.global_variables_initializer()
    self.sess.run(init)

    # train the model
    step, g_step, epoch = 0, 0, 0
    while mnist.train.epochs_completed < n_epoch:
    
      n_critic = 100 if g_step < 25 or (g_step+1) % 500 == 0 else self.n_critic

      start_time = time.time()
      for i in range(n_critic):
        losses_d = []

        # load the batch
        X_batch = mnist.train.next_batch(n_batch)[0]
        X_batch = X_batch.reshape((n_batch, 1, 28, 28))
        noise = np.random.rand(n_batch,100).astype('float32')
        feed_dict = self.load_batch(X_batch, noise)

        # train the critic/discriminator
        loss_d = self.train_d(feed_dict)
        losses_d.append(loss_d)

      loss_d = np.array(losses_d).mean()

      # train the generator
      # noise = np.random.rand(n_batch,100).astype('float32')
      noise = np.random.uniform(-1.0, 1.0, [n_batch, 100]).astype('float32')
      feed_dict = self.load_batch(X_batch, noise)
      loss_g = self.train_g(feed_dict)
      g_step += 1

      if g_step < 100 or g_step % 100 == 0:
        tot_time = time.time() - start_time
        print 'Epoch: %3d, Gen step: %4d (%3.1f s), Disc loss: %.6f, Gen loss %.6f' % \
          (mnist.train.epochs_completed, g_step, tot_time, loss_d, loss_g)

      # take samples
      if g_step % 100 == 0:
        noise = np.random.rand(n_batch,100).astype('float32')
        samples = self.gen(noise)
        samples = samples[:42]
        fname = logdir + '.mnist_samples-%d.png' % g_step
        plt.imsave(fname,
                   (samples.reshape(6, 7, 28, 28)
                           .transpose(0, 2, 1, 3)
                           .reshape(6*28, 7*28)),
                   cmap='gray')

      # saver.save(self.sess, checkpoint_root, global_step=step)

  def gen(self, noise):
    X_g_in, X_d_in = self.inputs
    feed_dict = { X_g_in : noise, K.learning_phase() : False }
    return self.sess.run(self.P, feed_dict=feed_dict)

  def train_g(self, feed_dict):
    _, loss_g = self.sess.run([self.train_op_g, self.loss_g], feed_dict=feed_dict)
    return loss_g

  def train_d(self, feed_dict):
    # clip the weights, so that they fall in [-c, c]
    self.sess.run(self.clip_updates, feed_dict=feed_dict)

    # take a step of RMSProp
    self.sess.run(self.train_op_d, feed_dict=feed_dict)

    # return discriminator loss
    return self.sess.run(self.loss_d, feed_dict=feed_dict)

  def train(self, feed_dict):
    self.sess.run(self.train_op, feed_dict=feed_dict)

  def load_batch(self, X_train, noise, train=True):
    X_g_in, X_d_in = self.inputs
    return {X_g_in : noise, X_d_in : X_train, K.learning_phase() : train}

  def eval_err(self, X, n_batch=128):
    batch_iterator = iterate_minibatches(X, n_batch, shuffle=True)
    loss_g, loss_d, p_real, p_fake = 0, 0, 0, 0
    tot_loss_g, tot_loss_d, tot_p_real, tot_p_fake = 0, 0, 0, 0
    for bn, batch in enumerate(batch_iterator):
      noise = np.random.rand(n_batch,100)
      feed_dict = self.load_batch(batch, noise)
      loss_g, loss_d, p_real, p_fake \
        = self.sess.run([self.d_real, self.d_fake, self.p_real, self.p_fake], 
                        feed_dict=feed_dict)
      tot_loss_g += loss_g
      tot_loss_d += loss_d
      tot_p_real += p_real
      tot_p_fake += p_fake
    return tot_loss_g / (bn+1), tot_loss_d / (bn+1), \
           tot_p_real / (bn+1), tot_p_fake / (bn+1)

# ----------------------------------------------------------------------------

def make_dcgan_discriminator(x, reuse=False):
  with tf.variable_scope('discriminator', reuse=reuse):
    bs = tf.shape(x)[0]
    x = tf.reshape(x, [bs, 28, 28, 1])
    x = tf.transpose(x, [0, 3, 1, 2])
    conv1 = tc.layers.convolution2d(
        x, 64, [4, 4], [2, 2],
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        activation_fn=tf.identity
    )
    conv1 = leaky_relu(conv1)
    conv2 = tc.layers.convolution2d(
        conv1, 128, [4, 4], [2, 2],
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        activation_fn=tf.identity
    )
    conv2 = leaky_relu(tc.layers.batch_norm(conv2))
    conv3 = tc.layers.convolution2d(
        conv2, 128, [4, 4], [1, 1],
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        activation_fn=tf.identity
    )
    conv3 = leaky_relu(tc.layers.batch_norm(conv3))
    conv3 = tcl.flatten(conv3)
    fc1 = tc.layers.fully_connected(
        conv3, 1024,
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        activation_fn=tf.identity
    )
    fc1 = leaky_relu(tc.layers.batch_norm(fc1))
    fc2 = tc.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
  return fc2

def make_dcgan_generator(z, reuse=False):
  with tf.variable_scope('generator'):
    bs = tf.shape(z)[0]
    fc1 = tc.layers.fully_connected(
        z, 1024,
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
        activation_fn=tf.identity
    )
    fc1 = tc.layers.batch_norm(fc1)
    fc1 = tf.nn.relu(fc1)
    fc2 = tc.layers.fully_connected(
        fc1, 7 * 7 * 128,
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
        activation_fn=tf.identity
    )
    fc2 = tf.reshape(fc2, tf.pack([bs, 7, 7, 128]))
    fc2 = tc.layers.batch_norm(fc2)
    fc2 = tf.nn.relu(fc2)
    conv1 = tc.layers.convolution2d_transpose(
        fc2, 64, [4, 4], [2, 2],
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
        activation_fn=tf.identity
    )
    conv1 = tc.layers.batch_norm(conv1)
    conv1 = tf.nn.relu(conv1)
    conv2 = tc.layers.convolution2d_transpose(
        conv1, 1, [4, 4], [2, 2],
        weights_initializer=tf.random_normal_initializer(stddev=0.02),
        weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
        activation_fn=tf.sigmoid
    )
    # conv2 = tf.reshape(conv2, tf.pack([bs, 784]))
    conv2 = tf.transpose(conv2, [0,3,1,2])
  return conv2
    
# ----------------------------------------------------------------------------
    
def make_keras_dcgan_discriminator(Xk_d):
  x = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, subsample=(2,2),
        activation=None, border_mode='same', init=conv2D_init,
        dim_ordering='th')(Xk_d)
  x = BatchNormalization(mode=2, axis=1)(x)
  x = LeakyReLU(0.2)(x)

  x = Convolution2D(nb_filter=128, nb_row=4, nb_col=4, subsample=(2,2),
        activation=None, border_mode='same', init=conv2D_init,
        dim_ordering='th')(x)
  x = BatchNormalization(mode=2, axis=1)(x)
  x = LeakyReLU(0.2)(x)

  # x = Convolution2D(nb_filter=128, nb_row=5, nb_col=5, subsample=(2,2),
  #       activation=None, border_mode='same', init=conv2D_init,
  #       dim_ordering='th')(x)
  # x = BatchNormalization(mode=2, axis=1)(x)
  # x = LeakyReLU(0.2)(x)

  x = Flatten()(x)
  x = Dense(1024, init=conv2D_init)(x)
  x = BatchNormalization(mode=2)(x)
  x = LeakyReLU(0.2)(x)

  d = Dense(1, activation=None)(x)

  return d

def make_keras_dcgan_generator(Xk_g, n_lat):
  n_g_hid1 = 1024 # size of hidden layer in generator layer 1
  n_g_hid2 = 128  # size of hidden layer in generator layer 2

  x = Dense(n_g_hid1, init=conv2D_init)(Xk_g)
  x = BatchNormalization(mode=2, )(x)
  x = Activation('relu')(x)

  x = Dense(n_g_hid2*7*7, init=conv2D_init)(x)
  x = Reshape((n_g_hid2, 7, 7))(x)
  x = BatchNormalization(mode=2, axis=1)(x)
  x = Activation('relu')(x)

  x = Deconvolution2D(64, 5, 5, output_shape=(128, 64, 14, 14), 
        border_mode='same', activation=None, subsample=(2,2), 
        init=conv2D_init, dim_ordering='th')(x)
  x = BatchNormalization(mode=2, axis=1)(x)
  x = Activation('relu')(x)

  g = Deconvolution2D(n_chan, 5, 5, output_shape=(128, n_chan, 28, 28), 
        border_mode='same', activation='sigmoid', subsample=(2,2), 
        init=conv2D_init, dim_ordering='th')(x)

  return g

# ----------------------------------------------------------------------------

def conv2D_init(shape, dim_ordering='tf', name=None):
  return initializations.normal(shape, scale=0.02, dim_ordering=dim_ordering, name=name)

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)  

def make_tweaked_discriminator(Xk_d):
  x = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(2,2),
        activation=None, border_mode='same', init=conv2D_init, bias=False,
        dim_ordering='th')(Xk_d)
  x = BatchNormalization(mode=2, axis=1)(x)
  x = LeakyReLU(0.2)(x)

  x = Convolution2D(nb_filter=128, nb_row=3, nb_col=3, subsample=(2,2),
        activation=None, border_mode='same', init=conv2D_init, bias=False,
        dim_ordering='th')(x)
  x = BatchNormalization(mode=2, axis=1)(x)
  x = LeakyReLU(0.2)(x)

  x = Convolution2D(nb_filter=1, nb_row=3, nb_col=3, subsample=(2,2),
        activation=None, border_mode='same', init='glorot_uniform',
        dim_ordering='th')(x)
  d = GlobalAveragePooling2D()(x)

  return d

def make_tweaked_generator(Xk_g, n_lat):
  s = 28
  f = 512

  x = Dense(f*7*7)(Xk_g)
  x = Reshape((f, 7, 7))(x)
  x = BatchNormalization(mode=2, )(x)
  x = Activation('relu')(x)

  x = UpSampling2D(size=(2,2))(x)
  nb_filters = 512 / 2
  x = Convolution2D(nb_filters, 3, 3,
        border_mode='same', activation=None,
        init=conv2D_init, dim_ordering='th')(x)
  x = BatchNormalization(mode=2, axis=1)(x)
  x = Activation('relu')(x)
  x = Convolution2D(nb_filters, 3, 3,
        border_mode='same', activation=None,
        init=conv2D_init, dim_ordering='th')(x)
  x = Activation('relu')(x)        

  x = UpSampling2D(size=(2,2))(x)
  nb_filters = 512 / 4
  x = Convolution2D(nb_filters, 3, 3,
        border_mode='same', activation=None,
        init=conv2D_init, dim_ordering='th')(x)
  x = BatchNormalization(mode=2, axis=1)(x)
  x = Activation('relu')(x)
  x = Convolution2D(nb_filters, 3, 3,
        border_mode='same', activation=None,
        init=conv2D_init, dim_ordering='th')(x)
  x = Activation('relu')(x)        

  g = Convolution2D(1, 3, 3,
        border_mode='same', activation='sigmoid',
        init=conv2D_init, dim_ordering='th')(x)

  return g