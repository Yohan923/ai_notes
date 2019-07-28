# Principal Component Analysis

Used for feature compression which can compress higher dimentional features into lower dimention features. This can be used to group highly correlated, speed up learning algorithm(only use PCA on training set and not cross validation set or test set.), compression for reducing disk space and data visualisation. Note: don't use PCA for dealing with overfitting since it doesn't use y values. Try use original x values with learning algorithms before planning to use PCA.

Intuition:

PCA tries to find the minimum value of the sum of projection errors from each point to the projected surface. That is the sum of the sqaure distance from points to the projected surface. PCA results in direction vectors $u^{(1)},...,u^{(k)}$ where $k$ is the k-dimention we are reducing to from n-dimention.

Data prefrocessing:

Always peform mean normalisation, and if features have very different scales, do feature scaling.

```Algorithm```:

1. compute covariance matrix:

$$
    \Sigma = \frac{1}{m} \sum_{i=1}^{n}(x^{(i)})(x^{(i)})^T
$$

2. compute left eigenvectors of $\Sigma$.
3. take first K eigen vectors.
4. compute $z^{(i)} = U_{reduced}^Tx^{(i)}$ for each example.

---
## Extras

### Reconstruction

We can reconstruct compressed representation back into the original with approximate values. Such that:

$$
    x_{approx} = U_{reduced}*z
$$

### Choosing K (number of principal component):

![PCA choose K](images\PCA_choose_k.jfif)

A better way of doing this instead of trying out different values of K on the above algorithm is to use the eigen values in diagnal form such that the matrix $S$ is a diagnal matrix with its diagnal elements as the eigen values. We can compute

$$
    1 - \frac{\sum^{k}_{i=1}S_{ii}}{\sum^{n}_{i=1}S_{ii}} \le 0.01
$$
with different values of k and use the smallest value of k that give the result.
