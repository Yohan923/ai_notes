# Anomaly Detection:

Finding outliers in training examples by using a trained model $p(x)$ for all training examples. For a new example we define $p(x_{test})<\epsilon$ to be an anomaly otherwise normal. Literally plotting gaussian distribution for the features and with new examples, we see if its features are very different from the mean. Similar to the percentile test.

## ```Algorithm```:

1. for training examples $\{x^{1},...,x^{m}\}$, fit parameters
    
    $$
        \begin{aligned}
        \mu_j &= \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)} \\

        \sigma^2_j &= \frac{1}{m} \sum_{i=1}^{m} (x_j^{(i)} - \mu_j)^2
        \end{aligned}
    $$
    mean can variande are fitted for each feature $j$.

2. given new example $x$, compute $p(x)$:

    $$
        \begin{aligned}
        p(x) &= \prod^{n}_{j=1} p(x_j;\mu_j,\sigma^2_j)\\

        p(x) &= \prod^{n}_{j=1} \frac{1}{\sqrt{2\pi}\sigma_j} e^{-\frac{(x_j-\mu_j)^2}{2\sigma^2_j}}
        \end{aligned}
    $$

## ```Evaluation```:

classification accuracy is not a good metrix since the data is very skewed(many normal examples and little anomalies).
Predict anomaly on cross validation set and find Precision and Recall([reference here](Evaluations_and_metrics.md)). Then Find $F_1$ score for evaluation.

$$  
    y = 
    \left\{
    \begin{aligned}
    &1 & if \ p(x)<\epsilon \\
    &0 & if \ p(x)\ge\epsilon
    \end{aligned}
    \right\}
$$

 We can also use cross validation set to find a good value for $\epsilon$ by taking the $\epsilon$ value with highest $F_1$ score.

---

## Extras

### ```Multivariate Gaussian distribution```

model $p(x)$ all in one go, $\mu$ and $\Sigma$ are covariance matrices.

![Multivariate gaussian distribution](images\Anomaly_multivariate.jfif)

Use this to correlate features automatically instead of needing to create new features in orginal model. A multivariate model with conturs being axis aligned is equivalent to the original model. 

- mutivariate model is much more expensive than original model
- must have larger number of examples than features such that $m>n$.
- $\Sigma$ may be non-invertible if there are too little features and lots of redundant featrues.


### Anomaly Vs Supervised:

Anomaly:
- large numbers of negative example, very low number of positive examples
- many different number types of anomalies
- future anomalies may be very different from what we have seen so far

Supervised:
- large number of negaive and positive examples.
- future positives are similar to what have learned
- large number of positive examples for algorithm to learn

### Choosing data:

- try to make sure data is gaussian, but even when not gaussian, algorithm may still work.
- can use transformations such as log(), square, square root etc.


