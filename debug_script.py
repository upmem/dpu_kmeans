# -*- coding: utf-8 -*-
import numpy as np

from dpu_kmeans import KMeans

X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
kmeans = KMeans(n_clusters=2, random_state=0, n_dpu=2, verbose=False).fit(X)

print(f"centroids: {kmeans.cluster_centers_}")
