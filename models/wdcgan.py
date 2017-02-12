import os
import time
import numpy as np
import keras.backend as K
import tensorflow as tf

import matplotlib.pyplot as plt

from keras.layers.convolutional import Convolution2D, Deconvolution2D, UpSampling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Reshape, Flatten, Activation
from keras.layers import Input
from keras.models import Model
from keras import initializations

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
      g = make_dcgan_generator(Xk_g, n_lat)
      # g = make_tweaked_generator(Xk_g, n_lat)

    # create discriminator
    with tf.name_scope('discriminator'):
      Xk_d = Input(shape=(n_chan, n_dim, n_dim))
      d = make_dcgan_discriminator(Xk_d)
      # d = make_tweaked_discriminator(Xk_d)

    # save inputs
    X_g = tf.placeholder(tf.float32, shape=(None, n_lat), name='X_g')
    X_d = tf.placeholder(tf.float32, shape=(None, n_chan, n_dim, n_dim), name='X_d')
    self.inputs = X_g, X_d

    # instantiate networks
    g_net = Model(input=Xk_g, output=g)
    d_net = Model(input=Xk_d, output=d)

    # get their weights
    self.w_g = [w for l in g_net.layers for w in l.trainable_weights]
    self.w_d = [w for l in d_net.layers for w in l.trainable_weights]

    # create predictions
    d_real = d_net(X_d)
    d_fake = d_net(g_net(X_g))
    self.P = g_net(X_g)

    # create losses
    self.loss_g = -tf.reduce_mean(d_fake)
    self.loss_d = -(tf.reduce_mean(d_real) - tf.reduce_mean(d_fake))

    # compute and store discriminator probabilities
    self.d_real = tf.reduce_mean(d_real)
    self.d_fake = tf.reduce_mean(d_fake)
    self.p_real = tf.reduce_mean(tf.sigmoid(d_real))
    self.p_fake = tf.reduce_mean(tf.sigmoid(d_fake))

    # create an optimizer
    lr = opt_params['lr']
    optimizer_g = tf.train.RMSPropOptimizer(lr)
    optimizer_d = tf.train.RMSPropOptimizer(lr)
    optimizer_d0 = tf.train.AdamOptimizer()

    # get gradients
    gv_g = optimizer_g.compute_gradients(self.loss_g, self.w_g)
    gv_d = optimizer_d.compute_gradients(self.loss_d, self.w_d)

    # create training operation
    self.train_op_g = optimizer_g.apply_gradients(gv_g)
    self.train_op_d = optimizer_d.apply_gradients(gv_d)
    self.train_op_d0 = optimizer_d0.apply_gradients(gv_d)
    self.train_op   = optimizer_d.apply_gradients(gv_d + gv_g)

    # clip the weights, so that they fall in [-c, c]
    self.clip_updates = [w.assign(tf.clip_by_value(w, -self.c, self.c)) for w in self.w_d]

  def fit(self, X_train, X_val, n_epoch=10, n_batch=128, logdir='dcgan-run'):
    # initialize log directory                  
    if tf.gfile.Exists(logdir): tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)

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
    step = 0
    g_step = 0
    for epoch in xrange(n_epoch):
      start_time = time.time()
      n_critic = 100 if g_step < 25 or (g_step+1) % 500 == 0 else self.n_critic
      
      minibatch_generator = iterate_minibatches(X_train, n_batch, shuffle=True)
      for nb, X_batch in enumerate(minibatch_generator):
        step += 1

        # load the batch
        noise = np.random.rand(n_batch,100).astype('float32')
        feed_dict = self.load_batch(X_batch, noise)

        # train the critic/discriminator
        loss_d = self.train_d(feed_dict, d0=False)

        # once every n_critic steps, train the generator as well
        if (nb+1) % n_critic == 0:
          g_step += 1
          noise = np.random.rand(n_batch,100).astype('float32')
          feed_dict = self.load_batch(X_batch, noise)
          self.train_g(feed_dict)
        
      # log results at the end of batch
      tr_g_err, tr_d_err, tr_p_real, tr_p_fake = self.eval_err(X_train)
      va_g_err, va_d_err, va_p_real, va_p_fake = self.eval_err(X_val)

      print "Epoch {} of {} took {:.3f}s ({} generator steps)".format(
        epoch + 1, n_epoch, time.time() - start_time, g_step)
      print "  training loss_g/loss_d/p_real/p_fake:\t\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
        tr_g_err, tr_d_err, tr_p_real, tr_p_fake)
      print "  validation loss_g/loss_d/p_real/p_fake:\t\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
        va_g_err, va_d_err, va_p_real, va_p_fake)

      # take samples
      samples = self.gen(np.random.rand(128, 100).astype('float32'))
      samples = samples[:42]
      fname = logdir + '.mnist_samples-%d.png' % epoch
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

  def train_d(self, feed_dict, d0=False):
    if not d0:
      # take a step of RMSProp
      self.sess.run(self.train_op_d, feed_dict=feed_dict)
    else:
      # take a step of Adam
      self.sess.run(self.train_op_d0, feed_dict=feed_dict)

    # clip the weights, so that they fall in [-c, c]
    self.sess.run(self.clip_updates, feed_dict=feed_dict)

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
# helpers

def iterate_minibatches(inputs, batchsize, shuffle=False):
  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
  for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
    if shuffle:
        excerpt = indices[start_idx:start_idx + batchsize]
    else:
        excerpt = slice(start_idx, start_idx + batchsize)
    yield inputs[excerpt]
    
# ----------------------------------------------------------------------------
    
def make_dcgan_discriminator(Xk_d):
  x = Convolution2D(nb_filter=64, nb_row=5, nb_col=5, subsample=(2,2),
        activation=None, border_mode='same', init=conv2D_init,
        dim_ordering='th')(Xk_d)
  x = BatchNormalization(axis=1)(x)
  x = LeakyReLU(0.2)(x)

  x = Convolution2D(nb_filter=64, nb_row=5, nb_col=5, subsample=(2,2),
        activation=None, border_mode='same', init=conv2D_init,
        dim_ordering='th')(Xk_d)
  x = BatchNormalization(axis=1)(x)
  x = LeakyReLU(0.2)(x)

  x = Convolution2D(nb_filter=128, nb_row=5, nb_col=5, subsample=(2,2),
        activation=None, border_mode='same', init=conv2D_init,
        dim_ordering='th')(x)
  x = BatchNormalization(axis=1)(x)
  x = LeakyReLU(0.2)(x)

  x = Flatten()(x)
  x = Dense(1024)(x)
  x = BatchNormalization()(x)
  x = LeakyReLU(0.2)(x)

  d = Dense(1, activation=None)(x)

  return d

def make_dcgan_generator(Xk_g, n_lat):
  n_g_hid1 = 1024 # size of hidden layer in generator layer 1
  n_g_hid2 = 128  # size of hidden layer in generator layer 2

  x = Dense(n_g_hid1)(Xk_g)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Dense(n_g_hid2*7*7)(x)
  x = Reshape((n_g_hid2, 7, 7))(x)
  x = BatchNormalization(axis=1)(x)
  x = Activation('relu')(x)

  x = Deconvolution2D(64, 5, 5, output_shape=(128, 64, 14, 14), 
        border_mode='same', activation=None, subsample=(2,2), 
        init=conv2D_init, dim_ordering='th')(x)
  x = BatchNormalization(axis=1)(x)
  x = Activation('relu')(x)

  g = Deconvolution2D(n_chan, 5, 5, output_shape=(128, n_chan, 28, 28), 
        border_mode='same', activation='sigmoid', subsample=(2,2), 
        init=conv2D_init, dim_ordering='th')(x)

  return g

# ----------------------------------------------------------------------------

def conv2D_init(shape, dim_ordering='tf', name=None):
    return initializations.normal(shape, scale=0.02, dim_ordering=dim_ordering, name=name)

def make_tweaked_discriminator(Xk_d):
  x = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(2,2),
        activation=None, border_mode='same', init=conv2D_init, bias=False,
        dim_ordering='th')(Xk_d)
  x = BatchNormalization(axis=1)(x)
  x = LeakyReLU(0.2)(x)

  x = Convolution2D(nb_filter=128, nb_row=3, nb_col=3, subsample=(2,2),
        activation=None, border_mode='same', init=conv2D_init, bias=False,
        dim_ordering='th')(x)
  x = BatchNormalization(axis=1)(x)
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
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = UpSampling2D(size=(2,2))(x)
  nb_filters = 512 / 2
  x = Convolution2D(nb_filters, 3, 3,
        border_mode='same', activation=None,
        init=conv2D_init, dim_ordering='th')(x)
  x = BatchNormalization(axis=1)(x)
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
  x = BatchNormalization(axis=1)(x)
  x = Activation('relu')(x)
  x = Convolution2D(nb_filters, 3, 3,
        border_mode='same', activation=None,
        init=conv2D_init, dim_ordering='th')(x)
  x = Activation('relu')(x)        

  g = Convolution2D(1, 3, 3,
        border_mode='same', activation='sigmoid',
        init=conv2D_init, dim_ordering='th')(x)

  return g