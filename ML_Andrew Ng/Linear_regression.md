# Linear Regression

Used for models where predicitons are made to determine a output with continuous values, such as housing prices.

Hypothesis has the structure as follows:

$$
    h_\theta(x) = \theta_0 + \theta_1x_1+...+\theta_nx_n
$$

thus in vectorised form $h$ can be written as:

$$
    h_\theta(x) = \theta^Tx
$$

where $n$ is the number of features in a multivariate linear regression model.

Cost function is given by:

$$
    J(\theta) = \frac {1}{2m} \sum_{i=1}^m (h_\theta(x^i)-y^i)^2
$$

This function is otherwise called the "Squared error function", or "Mean squared error", it denotes in simple terms the difference between predicted result and the actual result, $\frac 1 2$ is used simply for convenience in use with gradient descent.

Regularised cost function:

$$
    J(\theta) = \frac {1}{2m} \sum_{i=1}^m (h_\theta(x^i)-y^i)^2 +
    \lambda \sum_{j=1}^{n} \theta^2_j
$$

note: do not regularise $\theta_0$ in $\sum_{j=1}^{n} \theta^2_j$ this is done explicitly since $\theta$ index starts at 0

