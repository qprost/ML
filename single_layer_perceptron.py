# coding=utf-8
from abc import abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

from tqdm import tqdm


class SingleLayerNeuralNetwork:
    """
    ABC for single layer perceptron models
    Such an object is characterized by:
        - an activation function
        - a cost function
        - a predict function

    """

    def __init__(self, learning_rate=0.01, n_iter=100, seed=2018):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self._seed = seed
        self._weights = None
        self._cost = None

    @property
    def cost(self):
        return self._cost

    @property
    def weights(self):
        return self._weights

    def _initialize(self, X):
        """
        Initialize Perceptron weights (and error vector)
        :param X: (np.array) training data [n_samples, n_features]
        :return: None
        """
        n_samples, n_features = X.shape
        random_generator = np.random.RandomState(self._seed)
        self._weights = random_generator.normal(0, 0.01, size=n_features + 1)
        self._cost = np.nan * np.ones(self.n_iter)

    @staticmethod
    def _check_data(X, y):
        error_msg = 'Input and output must have the same length'
        assert y.shape[0] == X.shape[0], error_msg

    def net_input(self, X):
        """ Calculate dot product: <W, X>
        """
        return np.dot(X, self._weights[1:]) + self._weights[0]

    @abstractmethod
    def _fit(self, X, y):
        pass

    def fit(self, X, y):
        # Initialize weights
        self._initialize(X)
        self._check_data(X, y)
        # Train the perceptron
        print('Training {}'.format(repr(self)))
        self._fit(X, y)
        print('Training completed')

    @abstractmethod
    def _update(self, x, target):
        """
        Return error after weight update iteration
        """
        pass

    @abstractmethod
    def cost_function(self, y, y_hat):
        pass

    @abstractmethod
    def activation_function(self, x):
        pass

    def predict(self, X):
        """ Return predicted class label
        """
        return np.where(self.activation_function(X) >= 0.0, 1, -1)

    @staticmethod
    def heaviside(X):
        """ Heaviside function
        """
        return np.where(X >= 0.0, 1, -1)

    def plot(self, return_fig=False):
        """ Misclassification plot
        """
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(range(1, len(self._cost) + 1), self._cost, marker='o')
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Error")
        plt.title('Training - Convergence')
        ax[1].bar(range(len(self._weights)), self._weights)
        plt.title('Weights')
        plt.tight_layout()
        plt.show()
        if return_fig:
            return fig


class Perceptron(SingleLayerNeuralNetwork):
    """
    Simple Perceptron with unit-step activation function
    """

    def _fit(self, X, y):
        for i in tqdm(range(self.n_iter)):
            err = 0
            # looping through training sample components xi to update weights
            for xi, target in zip(X, y):
                err += self._update(xi, target)
            self._cost[i] = err

    def _update(self, x, target):
        """ Update weights using the Perceptron learning rule.
        """
        error = target - self.predict(x)
        delta_w = self.learning_rate * error
        # Bias unit
        self._weights[0] += delta_w
        # Input units
        self._weights[1:] += delta_w * x
        return self.cost_function(target, self.predict(x))

    def cost_function(self, y, y_hat):
        """Return classification error (belongs to {-2, 0, 2})"""
        error = y - y_hat
        return int(error != 0.0)

    def activation_function(self, x):
        return x


class Adaline(SingleLayerNeuralNetwork):
    """
    Linear single neuron network
    """

    def activation_function(self, x):
        return x

    def cost_function(self, y, y_hat):
        """SSE / L2 cost function"""
        error = y - y_hat
        return 0.5 * np.sum(error ** 2)

    def _fit(self, X, y):
        for i in tqdm(range(self.n_iter)):
            self._cost[i] = self._update(X, y)

    def _update(self, x, target):
        # Compute errors
        output = self.activation_function(self.net_input(x))
        # Batch gradient descent
        self._weights[1:] += self.learning_rate * np.dot(x.T, target - output)
        self._weights[0] += self.learning_rate * (target - output).sum()
        return self.cost_function(target, output)


class AdalineSGD(Adaline):
    """
    Stochastic gradient descent learning
    """

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _fit(self, X, y):
        for i in tqdm(range(self.n_iter)):
            # Create random training sample
            X, y = self._shuffle(X, y)
            self._cost[i] = self._update(X, y)


class LogisticRegression(Adaline):
    """
    Logistic Regression classifier
    """

    def activation_function(self, x):
        """Sigmoid function"""
        return 1.0 / (1.0 + np.exp(-x))

    def cost_function(self, y, y_hat):
        """Cross-entropy cost function"""
        return -y.dot(np.log(y_hat)) - (1. - y).dot(np.log(1. - y_hat))


if __name__ == '__main__':

    plot_data = False

    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                     header=None)

    # Train / test sample
    sample_len = len(df.index)
    cutoff = 0.75

    # Training sample
    y_train = np.where(
        df.iloc[0:int(np.round(cutoff * sample_len)), 4].values == 'Iris-setosa', -1, 1)
    X_train = df.iloc[0:int(np.round(cutoff * sample_len)), [0, 2]].values

    # Scatter plot
    if plot_data:
        plt.scatter(X_train[:50, 0], X_train[:50, 1],
                    color='red', marker='o', label='setosa')
        plt.scatter(X_train[50:100, 0], X_train[50:100, 1],
                    color='blue', marker='x', label='versicolor')
        plt.xlabel('petal length')
        plt.ylabel('sepal length')

    # Logistic Regression
    X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
    lr = LogisticRegression(learning_rate=0.05, n_iter=1000)
    lr.fit(X_train_01_subset, y_train_01_subset)
    lr.plot()
