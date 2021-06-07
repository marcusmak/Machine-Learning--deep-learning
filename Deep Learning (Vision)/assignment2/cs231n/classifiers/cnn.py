from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, batch_norm = "None" , bn_param = None):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        batch_modes = ["None","Layer","Batch","Instance"]
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        if(not batch_norm in batch_modes):
            batch_norm = "None"
        self.batch_norm = batch_norm
        if(batch_norm != "None"):
            self.bn_param = bn_param
            if(self.batch_norm == "Batch"):
                self.params['gamma'] = np.ones(num_filters)
                self.params['beta']  = np.zeros(num_filters)
            elif(self.batch_norm == "Layer"):
                self.params['gamma'] = None
                self.params['beta'] = None
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.params['W1'] = np.random.randn(num_filters,input_dim[0],filter_size,filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters)
        feat_dim = input_dim[1] * input_dim[2] // 4
        self.params['W2'] = np.random.randn(num_filters * feat_dim,hidden_dim) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.randn(hidden_dim,num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        if (self.batch_norm == "None"):
            out, cache1 = conv_relu_pool_forward(X, W1, b1 , conv_param, pool_param)
            out, cache2 = affine_relu_forward(out,W2,b2)
            out, cache3 = affine_forward(out, W3, b3)
            scores = out
        else:
            out, cache1 = conv_forward_fast(X, W1, b1 , conv_param)
            if(self.batch_norm == "Batch"):
                out, cache1a = spatial_batchnorm_forward(X,self.params['gamma'] , self.params['beta'] , self.bn_param) 
            elif(self.batch_norm == "Layer"):
                if (self.params['gamma'] is None):
                    self.params['gamma'] = np.ones(out.shape[0])
                    self.params['beta'] = np.zeros(out.shape[0])
                out, cache1a = layer_batchnorm_forward(X,self.params['gamma'] , self.params['beta'] , self.bn_param) 

            out, cache1b = relu_forward(out)
            out, cache1c = max_pool_forward_fast(out,pool_param)
            out, cache2 = affine_relu_forward(out,W2,b2)
            out, cache3 = affine_forward(out, W3, b3)
            scores = out
         
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        if (self.batch_norm == "None"):
            loss , dout = softmax_loss(scores,y)
            loss  += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3) )
            dx, grads['W3'] ,grads['b3'] = affine_backward(dout,cache3)
            dx, grads['W2'] ,grads['b2'] = affine_relu_backward(dx,cache2)
            dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, cache1)

            grads['W3'] += self.reg * W3
            grads['W2'] += self.reg * W2
            grads['W1'] += self.reg * W1
        else:
            loss , dout = softmax_loss(scores,y)
            loss  += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3) )
            dx, grads['W3'] ,grads['b3'] = affine_backward(dout,cache3)
            dx, grads['W2'] ,grads['b2'] = affine_relu_backward(dx,cache2)
            dx = max_pool_backward_fast(dx,cache1c)
            dx = relu_backward(dx,cache1b)
            if(self.batch_norm == "Batch"):
                dx, grads['gamma'], grads['beta'] = spatial_batchnorm_backward(dx,cache1a) 
            elif(self.batch_norm == "Layer"):
                dx, grads['gamma'], grads['beta'] = layer_batchnorm_backward(dx,cache1a) 

            dx, grads['W1'] ,grads['b1'] = conv_backward_fast(dx, cache1)
            grads['W3'] += self.reg * W3
            grads['W2'] += self.reg * W2
            grads['W1'] += self.reg * W1
            
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
