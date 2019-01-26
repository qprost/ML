import math
from typing import List, Union, Optional


class KMeans:
    """
    Pure Python implementation of K-means algorithm.
    """

    def __init__(self, n_clusters: int, data: List[List[Union[int, float]]]):
        self.n_clusters = n_clusters
        self.data = [[] for _ in range(n_clusters)]
        for elt in data:
            self.data[0].append(elt)
        self._means = None

    def cluster_data(self,
                     initial_means: List[List[Union[int, float]]],
                     n_iterations: Optional[int] = 10):
        self._means = initial_means
        self._assign_labels()

        i = 0
        while i < n_iterations:
            print("Running iteration: ", i)
            self._compute_means()
            self._assign_labels()
            i += 1

    def _compute_means(self) -> bool:
        res = True
        for k, cluster in enumerate(self.data):
            mean = [0, 0]
            for elt in cluster:
                mean[0] += elt[0]
                mean[1] += elt[1]
            mean[0] /= len(cluster)
            mean[1] /= len(cluster)

            res = (self._means[k] == mean) & res
            self._means[k] = mean
            print('Cluster #{} centroid: {}'.format(k, mean))

        return res

    def _assign_labels(self) -> None:
        new_data = [[] for _ in range(self.n_clusters)]

        for cluster in self.data:
            for elt in cluster:
                closest_mean = self._compute_closest_centroid(elt)
                new_data[closest_mean].append(elt)

        self.data = new_data

    def _compute_closest_centroid(self, point: List[Union[int, float]]) -> int:
        distances = []
        for mean in self._means:
            dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(point, mean)))
            distances.append(dist)

        return distances.index(min(distances))

    def print_clusters(self) -> None:
        for k, cluster in enumerate(self.data):
            print('Cluster #{}:\n'.format(k))
            for elt in cluster:
                print(elt, '\n')


if __name__ == '__main__':
    data = [[1.1, 1.],
            [1.4, 2.],
            [3.8, 7.],
            [4.3, 6.],
            [8., 5.8],
            [6., 8.5],
            [3., 2.],
            [9., 6.],
            [9.1, 4.]]

    # cluster1 = np.array([[3, 2], [1.4, 2], [1.1, 1]])
    # cluster2 = np.array([[8, 5.8], [6, 8.5], [9.1, 4], [9, 6]])
    # clusters = [cluster1, cluster2]

    km = KMeans(4, data)
    n_iter = 5
    init_means = [[1, 1], [3, 4], [8, 8], [2, 2]]
    km.cluster_data(init_means, n_iter)
    km.print_clusters()
