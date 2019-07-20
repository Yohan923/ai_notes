# Gradient Descent

Given starting parameter $\theta$,  gradient descent aims to find the optimal solution for $\theta$ by finding the minimum of the cost function $J(\theta)$.

The general form of gradient descent is given by:

$$
    \theta := \theta - \alpha \frac{\partial}{\partial\theta} J(\theta)
$$

the $\alpha$ is the learning rate, it should not be too big or too small. Large learning rate may result in over prediction which gradient descent may never converge. $\frac{\partial}{\partial\theta} J(\theta)$ is the partial derivitive. The intuition is that as gradient descent converges $\frac{\partial}{\partial\theta} J(\theta)$ will reach 0, therefore $\theta$ will have little or no change.


## Specifically:

### [Linear regression](Linear_regression):

repeat until convergence    
{  

$$
    \begin{aligned}

    \theta_0 &:= \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_\theta(x^i) - y^i) \\

    \theta_j &:= \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m}((h_\theta(x^i) - y^i)x_j^i) \\

    \end{aligned}
$$

}

where $j = 1...n$ where $n$ is the number of features

### [Logistic Regression](Logistic_regression):

repeat until convergence    
{   

$$
    \theta_j := \theta_j - \alpha\frac{1}{m} 
    \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
$$

}


## Regularisation

This is used to overcome overfitting which captures well the existing training examples but does not model prediction very well. Regularisation aims to reduce the magnitude of the prarameters $\theta$ except $\theta_0$, $\lambda$ is the regularization parameter. It determines how much the costs of our theta parameters are inflated.

### Specifically:

#### [Linear regression](Linear_regression):

repeat until convergence    
{  

$$
    \begin{aligned}

    \theta_0 &:= \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_\theta(x^i) - y^i) \\

    \theta_j &:= \theta_j - \alpha 
    [
        (\frac{1}{m} \sum_{i=1}^{m}(h_\theta(x^i) - y^i)x_j^i) +
        \frac{\lambda}{m}\theta_j
    ]\\

    \end{aligned}
$$

}

where $n$ is the number of features

#### [Logistic Regression](Logistic_regression):

repeat until convergence    
{   

$$
    \begin{aligned}
    \theta_0 &:= \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_\theta(x^i) - y^i) \\

    \theta_j &:= \theta_j - \alpha
    [
    (\frac{1}{m} 
    \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)})
    +
    \frac{\lambda}{m}\theta_j
    ]
    \end{aligned}
$$

}

---

## Extras

### Feature Scalling:

feature scalling and mean normalization can help to speed up by make features in similar, small value ranges. commonly use

$$
    x_i := \frac{x_i - \mu_i}{s_i}
$$

where $\mu$ is the mean of the feature and $s$ is the range of values(max - min).

### Advanced Optimisation:

- "Conjugate gradient"
- "BFGS"
- "L-BFGS" 

more sophisticated, faster ways to optimize θ that can be used instead of gradient descent. 

We will need a function to evaluate:

1. $\ J(\theta)$
2. $\frac{\partial}{\partial\theta_j} J(\theta)$

The use libraries to use the optimisation algorithms.
