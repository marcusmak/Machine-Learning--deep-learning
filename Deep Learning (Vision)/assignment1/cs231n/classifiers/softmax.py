import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  num_train = scores.shape[0]
  num_classes = scores.shape[1]
  for i in range(num_train): 
    scores -= scores[i].max(axis = 0) 
    loss_i = 0.0
    loss_i = np.sum(np.exp(scores[i]))
    for j in range(num_classes):
      dW[:,j] += X[i].T * np.exp(scores[i,j])/loss_i
    loss_i = np.exp(scores[i,y[i]]) / loss_i
    dW[:,y[i]] -= X[i].T
    loss_i = -1 * np.log(loss_i)
    loss += loss_i

  
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW = dW + 2*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
 
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  softmax = X.dot(W)
  num_train = softmax.shape[0]
  num_classes = softmax.shape[1]

  softmax -= np.amax(softmax, axis = 1)[:,np.newaxis]
  softmax = np.exp(softmax)
  softmax /= np.sum(softmax, axis =1)[:,np.newaxis]
  loss =(-1* np.log(softmax[range(num_train),y])).sum()

  softmax[range(num_train),y] -= 1
  dW = X.T.dot(softmax)
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW = dW + 2*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

