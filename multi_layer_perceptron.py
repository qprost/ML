# coding=utf-8
import numpy as np
from scipy.special import expit
import sys


class NeuralNetMLP:

    def __init__(self, n_output, n_features, n_hidden=30,
                 l1=0.0, l2=0.0, epochs=500, learning_rate=0.001,
                 momentum_rate=0.0, decrease_const=0.0, shuffle=True,
                 minibatches=1, seed=None):
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches
        self.seed = seed
        np.random.seed(seed)

    def _encode_labels(self, y, k):
        """Encode class label with onehot vectors"""
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        w1 = np.random.uniform(-1.0, 1.0, size=self.n_hidden * (self.n_features + 1))
        w1 = w1.reshape(self.n_hidden, self.n_features + 1)
        w2 = np.random.uniform(-1.0, 1.0, size=self.n_output * (self.n_hidden + 1))
        w2 = w2.reshape(self.n_output, self.n_hidden + 1)
        return w1, w2

    def _sigmoid(self, z):
        return expit(z)

    def _add_bias_unit(self, X, how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones(X.shape[0] + 1, X.shape[1])
            X_new[1:, :] = X
        else:
            raise AttributeError('how must be columns or row')
        return X_new

    def _feedforward(self, X, w1, w2):
        """Feedforward the input data through the network to
        generate an output"""
        # Input layer (add bias unit to it)
        a1 = self._add_bias_unit(X, how='column')
        # Compute net input (matrix form)
        z2 = w1.dot(a1.T)
        # Hidden layer
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2, how='column')
        # Output layer
        z3 = w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1, z2, a2, z3, a3

    def _L2_regression(slef):
        pass
