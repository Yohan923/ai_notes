# Style Transfer

- deeper layers of CNN are learning a larger portion of an image such that shallow layers maybe are learning lines, patterns while deepper layers are learning objects, people, texts.

We need:
- Content image $C$
- Style image $S$
- Generated image $G$

```Cost```:

$$
    J(G) = \alpha J_{content}(C, G) + \beta J_{style}(S,G)
$$

1. generate G randomly
2. use gradient descent (updating pixel values)

$$
    G := G - \frac{\partial}{\partial G}J(G)
$$

```Content cost```:

![**Content Cost**](images/Content_cost.jfif)

```Style cost```:

- define as the correlation between activations across channels.

![**style Cost**](images/Style_cost.jfif)

- it is better to compute the sum of the style cost for multiple different layers. 

$$
    J_{style}(S,G) = \sum_{l} \lambda^{[l]} J_{style}^{[l]}(S,G)
$$

