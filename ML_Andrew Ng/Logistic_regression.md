# Logistic Regression

Used for classification where the result is only a small number of discrete values, such as 1 or 0.

for binary classification problems, hypothesis is give by:

$$
    \begin{aligned}

    h_\theta(x) &= g(\theta^Tx) \\ 
    z &= \theta^Tx \\
    g(z) &= \frac{1}{1 + e^{-z}} \\

    \end{aligned} 
$$

$h_\theta(x)$ give the probability that a output is 1 therefore where $h_\theta(x) \geqq 0.5$ we have a result of 1. We can draw a dicision boundary separating the positive and negative results from the optimal gradients $\theta$.

Cost function is given by:

$$
    \begin{aligned}
    J(\theta) &= \frac{1}{m} \sum_{i=1}^{m} Cost(h_\theta(x^{(i)},y^{(i)}) \\

    Cost(h_\theta(x),y) &= -log(h_\theta(x)) \ &if \ y = 1 \\

    Cost(h_\theta(x),y) &= -log(1-h_\theta(x)) \ &if \ y = 0 \\
    \end{aligned}\\

    J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} 
    [
        y^{(i)}log(h_\theta(x^{(i)})) + 
        (1 - y^{(i)})log(1-h_\theta(x^{(i)}))
    ]

$$

Regularized cost function:

$$
    J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} 
    [
        y^{(i)}log(h_\theta(x^{(i)})) + 
        (1 - y^{(i)})log(1-h_\theta(x^{(i)}))
    ] +
    \frac{\lambda}{2m} \sum_{j=1}^{n} \theta^2_j
$$

note: do not regularise $\theta_0$ in $\sum_{j=1}^{n} \theta^2_j$ this is done explicitly since $\theta$ index starts at 0

---

## Extras

### Multiclass Classification: One-vs-all:
 classification of data when we have more than two categories. Instead of y = {0,1} we will expand our definition so that y = {0,1...n}.
We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.




