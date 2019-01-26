import numpy as np
from tqdm import tqdm


class ModelNotTrainedError(Exception):
    pass


class Perceptron:

    def __init__(self, n_iter=100, learning_rate=0.001):
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._weights = None
        self._errors = None

    @property
    def errors(self):
        return self._errors

    def weights(self):
        return self._weights

    @staticmethod
    def _check_data(y, X):
        error_msg = 'Input and output must have the same length'
        assert y.shape[0] == X.shape[0], error_msg

    def fit(self, y, X):
        self._check_data(y, X)
        # Initialise weights with zeros
        w = np.zeros(X.shape[1] + 1)
        errors = []
        for _ in tqdm(range(self._n_iter)):
            # Loop over examples to update weights
            for i in range(len(y)):
                # import ipdb; ipdb.set_trace()
                # Compute prediction
                y_hat = np.dot(w[1:], X[i, :]) + w[0]
                # Classification error
                error = y[i] - y_hat
                # Perceptron update rule
                w[1:] += self._learning_rate * error * X[i, :]
                w[0] += self._learning_rate * error * 1.0
            errors.append(error)
        self._weights = w
        self._errors = np.array(errors)
        return w

    def predict(self, X):
        if self._weights is None:
            raise ModelNotTrainedError("Must call fit() method")
        return np.dot(self._weights[1:], X[1:]) + X[0]


if __name__ == '__main__':
    X = np.array([[0., 0., -1], [1., 1., 0.], [-1., 0., 0.], [0., 1., 1.]])
    y = np.array([0, 1, 0, 1])
    pp = Perceptron(n_iter=10000)
    pp.fit(y, X)

    import matplotlib.pyplot as plt
    plt.plot(pp.errors)
    plt.title('Classification error')
    plt.show()