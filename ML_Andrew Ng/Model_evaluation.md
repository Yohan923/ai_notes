# Model Evaluation

selecting model using 3 sets of data:
- training set ~60%
- cross validation set ~20%
- test set ~20%

We can now calculate three separate error values for the three different sets using the following method:

1. Optimize the parameters in $\Theta$ using the training set for each polynomial degree.
2. Find the polynomial degree d with the least error using the cross validation set.
3. Estimate the generalization error using the test set with $Jtest(Θ(d))$, (d = theta from polynomial with lower error);
This way, the degree of the polynomial d has not been trained using the test set.

## Test error:
1. For linear regression: 

$$
Jtest(Θ)=\frac{1}{2m_{test}} \sum^{m_{test}}_{i=1}
(h_{\Theta}(x^{(i)}_{test})−y^{(i)}_{test})^2
$$

2. For classification - Misclassification error (aka 0/1 misclassification error):

$$
err(hΘ(x),y)=\left\{ 
    \begin{aligned}
    &1 \ &if \ h_{\Theta}(x)\ge0.5 \ and \ y=0\ or\ h_\Theta(x)<0.5\ and\ y=1 \\ 
    &0 &otherwise
    \end{aligned}
    \right\}
    
$$

This gives us a binary 0 or 1 error result based on a misclassification. The average test error for the test set is:

$$
Test Error=\frac{1}{m_{test}} \sum^{m_{test}}_{i=1}err(h_\Theta(x^{(i)}_{test}),y^{(i)}_{test})
$$
This gives us the proportion of the test data that was misclassified.

---
## Bias and Variance

- Bias problem(underfitting): $J_{train}(\Theta),J_{CV}(\Theta)$ are high, $J_{CV}(\Theta) \approx J_{train}(\Theta)$
- Variance problem(overfitting): $J_{train}(\Theta)$ is low, $J_{CV}(\Theta)$ >> $J_{train}(\Theta)$

Our decision process can be broken down as follows:

- Getting more training examples: Fixes high variance
- Trying smaller sets of features: Fixes high variance
- Adding features: Fixes high bias
- Adding polynomial features: Fixes high bias
- Decreasing $\lambda$: Fixes high bias
- Increasing $\lambda$: Fixes high variance.

## Regularisation:

to choose a good $\lambda$ we need to:

- make a list of $\lambda$ values to try out
- train the train set with each of the $\lambda$
- for each trained $\Theta$, calculate cross validation error ***without*** regularisation($\lambda=0$), choose $\lambda$ that produces the lowest cross validation error
- test with test set for generalisation

## Learning Curves:

A plot of training set size vs cross validation error. Train with $1...m$ increasing size of training set and calculate error with each case. Error could be calculated by average of i randomly selected training examples to yield better result.

- High Bias: 
    - low training set size: $J_{train}(\Theta)$ low, $J_{CV}(\Theta)$ high.
    - high training set size: $J_{train}(\Theta)$ and $J_{CV}(\Theta)$ are high, $J_{train}(\Theta) \approx J_{CV}(\Theta)$
- High Variance:
    - low training set size: $J_{train}(\Theta)$ low, $J_{CV}(\Theta)$ high.
    - high training set size: $J_{train}(\Theta)$ increases $J_{CV}(\Theta)$ decreases, $J_{train}(\Theta) < J_{CV}(\Theta)$ where difference is significant indicating low generalisation

