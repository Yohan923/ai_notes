# Neural Networks

Can be used instead of logistic regression for problems with large number of features that requires non-linear hypothesis. Does seem to have better acuracy than logistic regression even in linear cases.

## ```Model```:

![Neural Network Model](images\neural_network_model.png)

We essentialy use the same sigmoid function as in logistic regression:

$$
    g(z) = \frac{1}{1 + e^{-z}} \\
$$

Neural networks are represented in layers of units. These may include an input layer, output layer and a number of hidden layers in between. Units in input layer are features $x^{(i)}$, units in output layers are hypothesis $h_\Theta(x)$.

$a_i^{(j)}$ denotes "activation" of unit i in layer j

$\Theta^{(j)}$ denotes matrix of weights controlling function mapping from layer $j$ to layer $j+1$, essentially each row in $\Theta^{(j)}$ are the weights for each unit in layer $j$

A bias unit $a_0^{(j)}$ of value 1 is added to each propagating layer for calculation. Same idea as $x_o$.

## ```Forward Propagation```:

Use trained classifiers $\Theta$ to make predictions by propagating through the layers.

In general:

$$
    a_i^{(j)} = g(\Theta_{i0}^{(j-1)}a^{(j-1)}_0 + 
    \Theta_{i1}^{(j-1)}a^{(j-1)}_1 + ... +
    \Theta_{in}^{(j-1)}a^{(j-1)}_n)\\

    where \ n = No.Units \ in \ layer \ j-1
$$

Vectorize:

$$
    \begin{aligned}
    z^{(j)} &= \Theta^{(j-1)}a^{(j-1)}\\
    a^{(j)} &= g(z^{(j)})\\
    \end{aligned}
$$

## ```Back Propagation```:
Finding optimal $\Theta$

Some definitions first:
- $L$ = total number of layers in the network
- $s_{l}$ = number of units (not counting bias unit) in layer l
- $K$ = number of output units/classes

Cost Function:

$$
    J(\Theta) = -\frac{1}{m} \sum^{m}_{i=1} \sum^{K}_{k=1} [
        y_k^{(i)}log(h_\Theta(x^{(i)})_k) +
        (1-y_k^{(i)})log(1-h_\Theta(x^{(i)})_k)
    ] +
    \frac{\lambda}{m} \sum^{L-1}_{l=1} \sum^{s_l}_{i=1} \sum^{s_{l+1}}_{j=1} (\Theta^{(l)}_{j,i})^2
$$

The regularisation term is literally for all of the $\Theta$ terms except for bias terms.

The algorithm loop for each training example m{
1. compute forward propagation to compute $a^{(l)}$ for each layer 
2. using output compute $a^{(L)} = a^{(L)} - y^{(i)}$ (with bias)
3. compute $\delta^{(L-1)},...,\delta^{(2)}$ 
4. acumulate values into $\Delta^{(l)}_{i,j}$ for each layer (remove bias)

}

5. divide accumulated values by m and regularise if needed into $D^{(l)}_{i,j}$, such that $\frac{\partial J(\Theta)}{\partial \Theta^{(l)}_{i,j}} = D^{(l)}_{i,j}$.

In detail:

$$
    \begin{aligned}
    \delta^{(l)} &= (\Theta^{(l)})^T\delta^{l+1} .* g'(z^{(l)})\\
    
    note\  \ g'(z^{(l)}) &= a^{(l)}.*(1-a^{(l)}) \\

    \Delta^{(l)} &= \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T \\
    \end{aligned}
$$
<br>

$$
    \begin{aligned}
    \frac{\partial J(\Theta)}{\partial \Theta^{(l)}_{i,j}} &= 
    D^{(l)}_{i,j} = \frac{1}{m} \Delta^{(l)}_{i,j} \  
    &for \  j=0 \\
    
    \frac{\partial J(\Theta)}{\partial \Theta^{(l)}_{i,j}} &= 
    D^{(l)}_{i,j} = \frac{1}{m}\Delta^{(l)}_{i,j} + 
    \frac{\lambda}{m} \Theta^{(l)}_{i,j} \  
    &for \  j\ge1
    \end{aligned}
$$



---

## Extras

### Multiclass Classification:

![Neural Network Model](images\neural_network_multiclass.png)

essentially a neural network with multiple units in the output layer. The same number of units as the number of classifiers needed. Each classifier trained to recognise one class as in logistic regression.


### Gradient checking:
employed to check whether the gradient is correct, however really computationally expensive therefore should drop checking after using it to validated the algorithm.

$$
    \frac{\partial J(\Theta)}{\partial \Theta} \approx 
    \frac{J(\Theta + \epsilon)-J(\Theta + \epsilon)}{2\epsilon}
$$

A small value for $\epsilon$  such as $10^{-4}$, guarantees that the math works out properly. If the value for $\epsilon$ is too small, we can end up with numerical problems.


### Random Initialization:

Initializing all theta weights to zero does not work with neural networks. When we backpropagate, all nodes will update to the same value repeatedly. Instead we can randomly initialize our weights for $\Theta$.

![neural_network_random_initialization](images\neural_network_random_initialization.png)