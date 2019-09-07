# **Regularization**:
- Improves overfitting(High Variance problem) by parameter decay i.e. increase $\lambda$ more $w$ goes to 0

### **L2 Regularization**(weight decay):

#### Logistic Regression:
$\|w\|_2^2$ - squared euclidian norm.

$$
    \begin{aligned}
    \|w\|_2^2 &= \sum^{n_x}_{j=1} w_j^2 = w^Tw \\
    J(w,b) &= \frac{1}{m} \sum^{m}_{i=1} L(y,\hat{y}) + \frac{\lambda}{2m} \|w\|_2^2 \\
    \end{aligned}
$$

#### DNN:

$\|w^{[l]}\|_F^2$ - frobenius norm.

$$
    \begin{aligned}
    \|w\|_F^2 &= \sum^{n^{[l-1]}}_{i=1} \sum^{n^{[l]}}_{j=1} (w_{ij}^{[l]})^2 \\
    J(W,b) &= \frac{1}{m} \sum^{m}_{i=1} L(y,\hat{y}) + \frac{\lambda}{2m} \sum^{L}_{l=1} \|w^{[l]}\|_F^2 \\
    \end{aligned}
$$

### **L1 Regularization**:

$\|w\|_1 = \sum^{n_x}_{j=1} w_j$

Makes model sparse(more zeros in $w$)
L2 better in practice

### **Dropout Regularization**:

On each layer have a probability to drop a set of nodes. e.g. 0.5 chance to drop each of the nodes in a layer. This supposably makes the network not able to rely on a single feature so have to spread weights. Have similar effect of L2 Reg. 

- use different dropouts each iteration
- no dropout at test time, as dont want test to be random.
- can have different keep-props for different layers

Inverted dropout:

Divide $a^{[l]}$ by the keep node probability, prevents $Z$ from reducing since some nodes are dropped i.e. set to zero.

## **Data Augumentation**:

Reduces overfitting by synthesising more training examples.

## **Early Stopping**:

Imporves dev set error by stopping gradient descent early since dev set error might increase as you keep training your data. However, this prevents minimizing Cost, Orthogonalization convention is not met.