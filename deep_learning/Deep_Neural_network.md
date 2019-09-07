# Deep Neural Network

Parameters:

$$
    w, b
$$

Loss Function:

$$
    L(y,\hat{y}) = -(y\log{\hat{y}} + (1-y)\log{(1-\hat{y})})
$$

Cost Funtion:

$$
    \begin{aligned}
        J(W,b) &= \frac{1}{m} \sum^{m}_{i=1} L(y,\hat{y}) \\
        J(W,b) &= - \frac{1}{m} \sum^{m}_{i=1} (y\log{\hat{y}} + (1-y)\log{(1-\hat{y})}) \\
    \end{aligned}
$$

---

Definitions:

- $m$ - number of training examples
- $W$ - parameters to find
- $b$ - bias
- $L$ - total number of layers in the network
- $Z$ - preactivation
- $A^{[l]}$ - activation of layer $l$
- $\alpha$ - learining rate
- $g^{[l]}(Z)$ - activation function of layer $l$, sigmoid, tanh, relu, softmax etc.
- $g^{'[l]}(Z^{[l]})$ - derivative of activation function of layer $l$

Dimensions:

$$ 
    \begin{aligned}
    X &: (x, m) \\
    Y &: (y, m) \\
    W^{[l]} &: (n^{[l]}, n^{[l-1]}) \\
    b^{[l]} &: (n^{[l]}, 1)
    \end{aligned}
$$

Algorithm:

    initialise parameters and define model structure

    for n_iter times{

        for layers 1 to L{
            1. forward prop
            2. compute cost
            3. back prop
        }

        update gradient
    }

Forward Propagation(Vectorized):

$$
    \begin{aligned}
    Z^{[l]} &= W^{[l]}A^{[l-1]} + b^{[l]} \\
    A^{[l]} &= g^{[l]}(Z^{[l]})
    \end{aligned}
$$

Backward Propagation(Vectorized):

$$
    \begin{aligned}
    \frac{\partial{J}}{\partial{A^{[L]}}} &= dA^{[L]} = -(\frac{Y}{A^{[L]}} - \frac{1-Y}{1-A^{[L]}}) \\
    \frac{\partial{J}}{\partial{Z^{[l]}}} &= dZ^{[l]} = dA^{[l] }g^{'[l]}(Z^{[l]}) \\
    \frac{\partial{J}}{\partial{W^{[l]}}} &= dW^{[l]} = \frac{1}{m} dZ^{[l]} A^{[l-1]T} \\
    \frac{\partial{J}}{\partial{b^{[l]}}} &= db^{[l]} = \frac{1}{m} \sum^{m}_{i=1} dZ^{[l](i)} \\
    \frac{\partial{J}}{\partial{A^{[l-1]}}} &= dA^{[l-1]} = W^{[l]T}dZ^{[l]} \\
    \end{aligned}
$$

Gradient Descent Update:

$$
    \begin{aligned}
    W^{[l]} &= W^{[l]} - \alpha dW^{[l]} \\
    b^{[l]} &= b^{[l]} - \alpha db^{[l]}
    \end{aligned}
$$

---
## **Gradient Checking**:

employed to check whether the gradient is correct, however really computationally expensive therefore should drop checking after using it to validated the algorithm. 

```Doesn't work with dropout``` turn off dropout before using.

$$
    \frac{\partial J(\Theta)}{\partial \Theta} \approx 
    \frac{J(\Theta + \epsilon)-J(\Theta + \epsilon)}{2\epsilon}\\ 
    check \  \frac{\|d\Theta_{approx} -d\Theta\|_2}{\|d\Theta_{approx}\|_2 - \|d\Theta\|_2}
$$

$\Theta$ is the comination of $W,b$. A small value for $\epsilon$  such as $10^{-7}$. If the value for $\epsilon$ is too small, we can end up with numerical problems.

## **Softmax**:

used for multi-class classification

![**softmax**](images\Softmax.jfif)

![**softmax loss**](images\Softmax_out.jfif)
