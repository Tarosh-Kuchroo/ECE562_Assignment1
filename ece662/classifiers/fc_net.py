from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):

        self.params = {}
        self.reg = reg

        #W1, b1
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        #W2, b2
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        #Forward: Affine then ReLU then affine
        h, cache1 = affine_relu_forward(X, W1, b1) #This is the hidden activation
        scores, cache2 = affine_forward(h, W2, b2) #This is the output class scores

        #Testting:
        if y is None:
            return scores
        
        #Loss = softmax loss + L2 regularization
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = data_loss + reg_loss

        #Backward
        grads = {}
        dh, dW2, db2 = affine_backward(dscores, cache2) #gradients from output layer
        dX, dW1, db1 = affine_relu_backward(dh, cache1) #gradients from hidden layer

        #this is to add regularization to the gradients
        dW2 += self.reg * W2
        dW1 += self.reg * W1

        #Packing gradients
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2 
        return loss, grads



class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        layer_dims = [input_dim] + list(hidden_dims) + [num_classes]

        for idx in range(1, len(layer_dims)):
            w_key = f"W{idx}"
            b_key = f"b{idx}"
            self.params[w_key] = weight_scale * np.random.randn(
                layer_dims[idx - 1], layer_dims[idx]
            )
            self.params[b_key] = np.zeros(layer_dims[idx])

        if self.normalization in {"batchnorm", "layernorm"}:
            for idx in range(1, len(layer_dims) - 1):
                gamma_key = f"gamma{idx}"
                beta_key = f"beta{idx}"
                self.params[gamma_key] = np.ones(layer_dims[idx])
                self.params[beta_key] = np.zeros(layer_dims[idx])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = X
        layer_caches = []

        for idx in range(1, self.num_layers):
            w = self.params[f"W{idx}"]
            b = self.params[f"b{idx}"]
            out, fc_cache = affine_forward(out, w, b)

            norm_cache = None
            if self.normalization == "batchnorm":
                gamma = self.params[f"gamma{idx}"]
                beta = self.params[f"beta{idx}"]
                out, norm_cache = batchnorm_forward(out, gamma, beta, self.bn_params[idx - 1])
            elif self.normalization == "layernorm":
                gamma = self.params[f"gamma{idx}"]
                beta = self.params[f"beta{idx}"]
                out, norm_cache = layernorm_forward(out, gamma, beta, self.bn_params[idx - 1])

            out, relu_cache = relu_forward(out)

            dropout_cache = None
            if self.use_dropout:
                out, dropout_cache = dropout_forward(out, self.dropout_param)

            layer_caches.append(
                {
                    "fc": fc_cache,
                    "norm": norm_cache,
                    "relu": relu_cache,
                    "dropout": dropout_cache,
                }
            )

        w = self.params[f"W{self.num_layers}"]
        b = self.params[f"b{self.num_layers}"]
        scores, final_cache = affine_forward(out, w, b)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        data_loss, dscores = softmax_loss(scores, y)
        loss = data_loss
        grads = {}

        for idx in range(1, self.num_layers + 1):
            w = self.params[f"W{idx}"]
            loss += 0.5 * self.reg * np.sum(w * w)

        dout, dW, db = affine_backward(dscores, final_cache)
        grads[f"W{self.num_layers}"] = dW + self.reg * self.params[f"W{self.num_layers}"]
        grads[f"b{self.num_layers}"] = db

        for idx in range(self.num_layers - 1, 0, -1):
            layer_cache = layer_caches[idx - 1]
            dropout_cache = layer_cache["dropout"]

            if self.use_dropout:
                dout = dropout_backward(dout, dropout_cache)

            dout = relu_backward(dout, layer_cache["relu"])

            if self.normalization == "batchnorm":
                dout, dgamma, dbeta = batchnorm_backward(dout, layer_cache["norm"])
                grads[f"gamma{idx}"] = dgamma
                grads[f"beta{idx}"] = dbeta
            elif self.normalization == "layernorm":
                dout, dgamma, dbeta = layernorm_backward(dout, layer_cache["norm"])
                grads[f"gamma{idx}"] = dgamma
                grads[f"beta{idx}"] = dbeta

            dout, dW, db = affine_backward(dout, layer_cache["fc"])
            grads[f"W{idx}"] = dW + self.reg * self.params[f"W{idx}"]
            grads[f"b{idx}"] = db

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
