import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from random import seed, randint
from sklearn.datasets import make_blobs

# create anisotropicly distributed data (Good for Mahalonobis, Bad for Euclidean)
# Dataset was taken from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py
data, labels = make_blobs(n_samples=500, random_state=170)
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
maha_data = (np.dot(data, transformation)).tolist()
euclid_data = (np.loadtxt("cluster_validation_data.txt", delimiter=","))
maha_seed = 265
euclid_seed = 317

def plot_cluster(clusters, title="", pos=111):
    plt.subplot(pos)
    plt.title(title)
    for C in clusters:
        C = np.array(C)
        plt.scatter(C[:, 0], C[:, 1])

def euclidean_distance(x, mu, *args):
    return np.linalg.norm(np.subtract(x,mu))

def mahalanobis_distance(x, mu, C):
    if len(C) < 2:
        return euclidean_distance(x, mu)
    COV = np.cov(np.transpose(np.subtract(C,mu)))
    S = np.linalg.pinv(COV)
    a = np.subtract(x, mu)
    return np.sqrt(np.dot(np.dot(a, S), a.T))

def kmeans(x, k=3, max_iters=10, random_state=1, dist=mahalanobis_distance):
    t = 0
    seed(random_state)
    mu = [x[randint(0, len(x) - 1)] for _ in range(k)]
    P =  [np.array([mu[i]]) for i in range(k)]
    for t in range(max_iters):
        P_t = deepcopy(P)
        for xj in x:
            index = np.argmin([dist(xj, mu[i], P_t[i]) for i in range(k)])
            if ([xj] == P_t[index]).all():
                P_t[index] = np.append(P_t[index], [xj], axis=0)                    
        mu = [np.mean(P_t[i], axis=0) for i in range(k)]

        if all([np.array_equal(P[i], P_t[i]) for i in range(k)]):
            print(f"Took {t} iters")
            return P_t

        P = P_t
    return P

def davies_bouldin(C):
    k = len(C)
    mu = [np.mean(c) for c in C]

    distances = [
        [euclidean_distance(data_point, mu[i]) for data_point in C[i]] for i in range(k)
    ]
    avg_dist = [np.mean(distances[i]) for i in range(k)]

    compute_Dij = lambda i, j: (avg_dist[i] + avg_dist[j]) / euclidean_distance(
        mu[i], mu[j]
    )
    d_ij = [compute_Dij(i, j) for i in range(k) for j in range(k) if i != j]
    return np.max(d_ij) / k

# Gather data for kmeans comparisons
good_maha = kmeans(maha_data, random_state=maha_seed)
bad_euclid = kmeans(maha_data, dist=euclidean_distance, random_state=maha_seed)
bad_maha = kmeans(euclid_data, random_state=euclid_seed)
good_euclid = kmeans(euclid_data, dist=euclidean_distance, random_state=euclid_seed)
plot_cluster(good_maha, title="Mahalobis", pos=221)
plot_cluster(bad_euclid, title="Euclidean", pos=222)
plot_cluster(bad_maha, pos=223)
plot_cluster(good_euclid, pos=224)
plt.show()

# Plot the Davies-Bouldin Results for Analysis
davies_results = []
for i in range(2, 7):
    clusters = kmeans(
        euclid_data, k=i, dist=euclidean_distance, random_state=72
    )
    if i < 6: # we only plot 4 of the K=i as an example
        plot_cluster(clusters, title=f"K={i}", pos=219 + i)
    davies_results.append(davies_bouldin(clusters))
plt.show()

plt.plot(range(2, 7), davies_results)
plt.xlabel("Number of clusters(k)")
plt.ylabel("Davies-Bouldin Values")
plt.show()
