# Convolutional Neural Network

## Fundamentals:

- Parameter Sharing
- sparse connection between input and output.

Taking convolution of data and filter!:
    
1. overlapping filter with data strating from top left.
2. take product of the overlapping values and add together all of the products to get a value in the resulting matrix
3. take a stride step and repeat all above until no more steps can be taken

- 'valid' convolution - no padding
- 'same' convolution - output size = input size.

<br>

- input $n * n * c$
- filter $f * f * c$
- padding $p$ (p=1 means 1 padding on each side)
- stride $s$ (step size) 

Output dimention:

$$
    \frac{n+2p-f}{s}+1
$$

round down if this is not integer

---

## Conv Layer:

- Condenses input size. Final Conv output is unrolled into a single vector and fed to a logistic/softmax regression NN.

A single layer looks like: 

![**Single layer**](images/Conv_net_layer.jfif)
![**Single layer dims**](images/Conv_net_layer_dims.jfif)

## Pooling layer:

### Max pooling:

- found to be useful in research, no one understand it

![**Single layer dims**](images/Max_pooling.jfif)

### Average pooling:

- used less than max pooling
- literally same as max pooling but operation is taking average.

## Fully connected layer:

- just like a neural network layer.

---

## Classic Nets:

### LeNet-5:

- trained on greyscale images
- used to recognize text in images
- used Sigmoid/tanh activations instead of Relu

![**LeNet-5**](images/LeNet_5.jfif)

### AlexNet:

- much larger than LeNet-5
- used Relu activation

![**AlexNet**](images/AlexNet.jfif)

### VGG-16:

- structure is very simple, but very large.

![**VGG-16**](images/VGG_16.jfif)

## ResNets:

- Good to build very deep networks
- use Residual blocks, activation can follow a shortcut/skip layer
- in practice, deep networks train error rises as network goes deeper, but with ResNet residual blocks it does'nt, helps with exploding/vanishing gradient.
- its easier to calculate identity function even when you add more layers. ie. you can get $a^{[l]}$ easily from $a^{[l+2]}$
- does'nt hurt performance
- same convs to make sure addition have same dimention
- if not same dimention, multiply by a matirx $W_s$.

Residual block:

$$
    a^{[l+2]} = g(z^{[l+2]} + a^{[l]})
$$

![**ResNet**](images/ResNet.jfif)

## Inception Net:

- do them all lmao and concatenate them.
- computational cost is high
- we can reduce comptational cost of 5x5 layer by using 1x1 conv as bottle neck before applying 5x5 to reduce the number of channels thus save computation.

![**Inception module**](images/Inception_module.jfif)

- side branches seems to have regularizing effect
- they helps to ensure features computed in hidden units are not to bad to compute the output

![**Inception net**](images/Inception_net.jfif)

## 1x1 convolutions:

- looks at one element of all of the channels.
- use to shrink number of channels.

## 1D and 3D data:

just generalise from 2D data. ensure number of channel is same for input and filters.