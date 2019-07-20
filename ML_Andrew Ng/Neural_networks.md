# Neural Networks

Can be used instead of logistic regression for problems with large number of features that requires non-linear hypothesis. Does seem to have better acuracy than logistic regression even in linear cases.

## Model:

![Neural Network Model](images\neural_network_model.png)

We essentialy use the same sigmoid function as in logistic regression:

$$
    g(z) = \frac{1}{1 + e^{-z}} \\
$$

Neural networks are represented in layers of units. These may include an input layer, output layer and a number of hidden layers in between. Units in input layer are features $x^{(i)}$, units in output layers are hypothesis $h_\Theta(x)$.

$a_i^{(j)}$ denotes "activation" of unit i in layer j

$\Theta^{(j)}$ denotes matrix of weights controlling function mapping from layer $j$ to layer $j+1$, essentially each row in $\Theta^{(j)}$ are the weights for each unit in layer $j$

A bias unit $a_0^{(j)}$ of value 1 is added to each propagating layer for calculation. Same idea as $x_o$.

## Forward Propagation

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

---

## Extras

### Multiclass Classification:

![Neural Network Model](images\neural_network_multiclass.png)

essentially a neural network with multiple units in the output layer. The same number of units as the number of classifiers needed. Each classifier trained to recognise one class as in logistic regression.