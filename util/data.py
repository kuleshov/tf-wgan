import sys
import os
import pickle
import tarfile

import numpy as np

# ----------------------------------------------------------------------------

def load_cifar10():
  """Download and extract the tarball from Alex's website."""
  dest_directory = '.'
  DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    if sys.version_info[0] == 2:
      from urllib import urlretrieve
    else:
      from urllib.request import urlretrieve

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)  

  def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
      datadict = pickle.load(f)
      X = datadict['data']
      Y = datadict['labels']
      X = X.reshape(10000, 3, 32, 32).astype("float32")
      Y = np.array(Y, dtype=np.uint8)
      return X, Y

  xs, ys = [], []
  for b in range(1,6):
    f = 'cifar-10-batches-py/data_batch_%d' % b
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch('cifar-10-batches-py/test_batch')
  return Xtr, Ytr, Xte, Yte

def load_mnist():
  # We first define a download function, supporting both Python 2 and 3.
  if sys.version_info[0] == 2:
    from urllib import urlretrieve
  else:
    from urllib.request import urlretrieve

  def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print "Downloading %s" % filename
    urlretrieve(source + filename, filename)

  # We then define functions for loading MNIST images and labels.
  # For convenience, they also download the requested files if needed.
  import gzip

  def load_mnist_images(filename):
    if not os.path.exists(filename):
      download(filename)
      # Read the inputs in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    data = data.reshape(-1, 1, 28, 28)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)

  def load_mnist_labels(filename):
    if not os.path.exists(filename):
      download(filename)
      # Read the labels in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data

  # We can now download and read the training and test set images and labels.
  X_train = load_mnist_images('train-images-idx3-ubyte.gz')
  y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
  X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
  y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

  # We reserve the last 10000 training examples for validation.
  X_train, X_val = X_train[:-10000], X_train[-10000:]
  y_train, y_val = y_train[:-10000], y_train[-10000:]

  # We just return all the arrays in order, as expected in main().
  # (It doesn't matter how we do this as long as we can read them again.)
  return X_train, y_train, X_val, y_val, X_test, y_test

# ----------------------------------------------------------------------------
# other

def load_noise(n=100,d=5):
  """For debugging"""
  X = np.random.randint(2,size=(n,1,d,d)).astype('float32')
  Y = np.random.randint(2,size=(n,)).astype(np.uint8)

  return X, Y

def load_h5(h5_path):
  """This was untested"""
  import h5py
  # load training data
  with h5py.File(h5_path, 'r') as hf:
    print 'List of arrays in input file:', hf.keys()
    X = np.array(hf.get('data'))
    Y = np.array(hf.get('label'))
    print 'Shape of X: \n', X.shape
    print 'Shape of Y: \n', Y.shape

    return X, Y

def whiten(X_train, X_valid):
  offset = np.mean(X_train, 0)
  scale = np.std(X_train, 0).clip(min=1)
  X_train = (X_train - offset) / scale
  X_valid = (X_valid - offset) / scale
  return X_train, X_valid    