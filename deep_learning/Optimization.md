# Optimization

## **Speeding Up Training**:

### **Normalising Input**:

create a more or less symmetric training curve, speed up gradient descent steps.

1. subtract from inputs, the mean $\mu$ of the inputs.
2. divide by the varience $\sigma^2$

normalise test set and training set using the same $\mu$ and $\sigma^2$

### **Vanishing/Exploding gradient**:

partial solution:

weight initialisation:

$$
    Var(W^{[l]}) = \frac{1}{n^{[l-1]}} \\
    W^{[l]} = np.random.randn(shape) * sqrt(Var(W^{[l]}))
$$

for Relu: $Var(W^{[l]}) = \frac{2}{n^{[l-1]}}$

for tanh(Xavier initialisation): $Var(W^{[l]}) = \sqrt{\frac{1}{n^{[l-1]}}}$

This initialises weights to be not too much lower or higher than 1 thus slowing down the exploding/vanishing gradient problem.

---

### **Mini-batch**:

Choose a small set of large input set for each epoch. Such that each epoch is composed of gradient descent on many mini-batches. Parameters are updated with each mini-batch thus we will get result more frequent than having to load the whole data set first. Stochastic(mini-batch of 1) lose vectorization speedup. Make mini-batch fit in memory. Mini-batch don't converge to a minimum but ocilates around a minimum.

## **Exponentially weighted average**:

$$
    V_{\theta_n} = \beta V_{\theta_{n-1}} + (1-\beta)\theta_n
$$

Bias correction:

Since the first few variables are just starting off, they would be much less than expected. 

$$
    V_{\theta_n} = \frac{\beta V_{\theta_{n-1}} + (1-\beta)\theta_n}{1-\beta^n}
$$

As $n$ becomes large, bias correction will become less relevant

## **Gradient Descent with momentum**:

Smooth out steps of gradient descent, meaning each step has less occilation. e.g. highly occilating direction will have negative and positive values, which sums close to zero, smoothing out occilation, while more positive values sums to higher posivitive value gaining more momentum. Gaining more speed in the desired direction.

In general $\beta = 0.9$ works ok.

bias correction is not usually implemented as should warm up relatively quick so no need.

On each mini-batch:

$$
    V_{dW} = \beta V_{dW} + (1-\beta)dW \\
    V_{db} = \beta V_{db} + (1-\beta)db \\
$$

then update parameters:

$$
    W = W - \alpha V_{dW} \\
    b = b - \alpha V_{db}
$$

## **RMSprop**:

On each mini-batch:

$$
    S_{dW} = \beta S_{dW} + (1-\beta)dW^2 \\
    S_{db} = \beta S_{db} + (1-\beta)db^2 \\
$$

then update parameters:

$$
    W = W - \alpha \frac{dW}{\sqrt{S_{dW}} + \epsilon} \\
    b = b - \alpha \frac{db}{\sqrt{S_{db}} + \epsilon}
$$

$\epsilon$ is very small, used to prevent from divide by 0

## **Adam**:

1. On each mini-batch t:

$$
    \begin{aligned}
    V_{dW} &= \beta_1 V_{dW} + (1-\beta_1)dW \\
    V_{db} &= \beta_1 V_{db} + (1-\beta_1)db \\
    S_{dW} &= \beta_2 S_{dW} + (1-\beta_2)dW^2 \\
    S_{db} &= \beta_2 S_{db} + (1-\beta_2)db^2 \\
    \end{aligned}
$$

2. Apply bia correction:

$$
    \begin{aligned}
    V_{dW} &= \frac{V_{dW}}{1-\beta_1^t} \\
    V_{db} &= \frac{V_{db}}{1-\beta_1^t} \\
    S_{dW} &= \frac{S_{dW}}{1-\beta_2^t} \\
    S_{db} &= \frac{S_{db}}{1-\beta_2^t} \\
    \end{aligned}
$$

3. then update parameters:

$$
    \begin{aligned}
    W &= W - \alpha \frac{V_{dW}}{\sqrt{S_{dW}} + \epsilon} \\
    b &= b - \alpha \frac{V_{db}}{\sqrt{S_{db}} + \epsilon} \\
    \end{aligned}

$$

$\beta_1 = 0.9$ $\beta_2 = 0.999$ $\epsilon = 10^{-8}$

---

### Learning Rate Decay:

Slowly lowering your learning rate, as rate decreases, the steps will be smaller as you approch convergence. Can also lower alpha by hand if you are bored.

1. 
$$
    \alpha = \frac{1}{1 + decayRate * epochNum} * \alpha_0 
$$

2. 

$$
    \alpha = 0.95^{epochNum} \alpha_0
$$

3. 

$$
    \alpha = \frac{k}{\sqrt{epochNum}} \alpha_0
$$

### Local Optima:

It turns out local optima isn't much of a problem but rather saddle point is more a problem. Scuh that we reach a plateu where the slope is 0.