# Face Recognition

## One shot learning:

- learn similarity functions which measures the degree of difference between images $d(img1,img2)$

## Siamese network

- converts the input image into an 'encoding' of vector of 128 numbers

then:

$$
    d(img1,img2) = \|f(img1) - f(img2)\|^2_2
$$

## Triplet loss:

- given Anchor, positive and negative images
- want distance between anchor and positive to be low, anchor and negative to be low

$$
    d(A,P) - d(A,N) + \alpha \le 0
$$

- $\alpha$ is the margin parameter which prevents algorithm from setting $d(A,P)$ and $d(A,N)$ all to zero, pushs the distance between tham further

Loss Function:

$$
    L(A,P,N) = max(d(A,P) - d(A,N)+\alpha, \ 0)
$$

- choose A,P,N randomly is every easy to fill the criteria therefore it won't learn much
- choose that are hard to train such that $d(A,P)$ is close to $d(A,N)$

## Face Rec and Binary Classification:

- use Binary classification on a pair of images and learn if they are different or same
- can precompute image encodings to be stored instead of computing encodings everytime

![**Binary Face**](images/Face_binary.jfif)

