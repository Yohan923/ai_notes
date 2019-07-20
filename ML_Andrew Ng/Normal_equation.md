# Normal Equation

$X$ is a matrix of the training set, such that each row of $X$ is the features of a training example with first column corresponding to $\theta_0$ of all $1s$. 

- No need to choose $\alpha$
- No need to iterate
- $O(n^3)$ complexity as need calculate inverse of $X^TX$
- Slow if $n$ is very large


## Linear regression:

$$
    (X^TX)^{-1}X^Ty
$$

Regularisation:

$$
    (X^TX+ \lambda L)^{-1}X^Ty
$$

where L is a matrix which differs from identity matrix only by the left top corner being 0.