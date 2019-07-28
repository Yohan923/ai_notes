# K-means

A cluster algorithm used to find clusters of data in unlabled training data.

Terms:
- $K$ - the number of clusters centroid.
- $\mu_k$ - average of the points assigned to cluster $k$
- $c^{(i)}$ - index of the cluster that $x^{(i)}$ is assigned to

Optimisation objective(cost):

$$
    min \ J(c^{(1)},...,c^{(m)},\mu_1,...,\mu_K) = 
    \frac{1}{m} \sum^{m}_{i=1} 
    ||x^{(i)}-\mu_{c^{(i)}}||^2
$$

In general we are trying to minimise the square distance from each training examples to their assigned cluster centroid.

```Algorithm```:

repeat until max_iter or convergence:

1. assign to each training example $x^{(i)}$ a centroid that is closest to it with index $c^{(i)}$ that signifies the centroid $\mu_k$.

2. find the mean of points assigned to each cluster centroid as $\mu_k$.

---
## Extras:

### Random initialisation:

Better to randomly choose K training examples to be the initial centroids. However you might randomly choose the clusters that leads to bad local minimas. Thus it is recommended to run K-means multiple times with different random initial clusters, compute the cost function(distortion), and choose the one that gives lowest cost. ```Note``` when K is large there might not be enough value to do random initialisation multiple times.

### Choosing K:

There is'nt really a best way, perhaps manually visualizing data and choose manually is still better. Do take into account the context of the training data and the practical use of the algoritm being implemented.

elbow method:

plot cost $J$ against number of K and there may be a 'elbow' on the curve so you choose K on the elbow. There might not be a clear answer so don't put much fait on it.