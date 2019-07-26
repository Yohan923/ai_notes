# Support Vector Machines

$$
    min_{\theta} \  C\sum_{i=1}^{m} [
        y^{(i)}cost_1(\theta^Tx^{(i)}) +
        (1 - y^{(i)})cost_0(\theta^Tx^{(i)})
        ] +
        \frac{1}{2} \sum_{i=1}^{n}\theta_j^2 \\

    C=\frac{1}{\lambda}
$$

Hypothesis:

$$
    h_\theta(x) 
    \left\{
    \begin{aligned}
            &1 &if \ \Theta^Tx \ge 0\\
            &0  &otherwise
    \end{aligned}
    \right\}
$$

![SVM](\images\SVM.jfif)

When C is large, SVM will try to fit every example perfectly thus including outliers leading to a unnatural boundary.

## ```Large Margin Classification```:

As we can see extra safety is given where $-1<\theta^Tx<1$ which gives the margin of decision boundary that separates the classes ie. larger minimum distance from boundary to any of the examples.


### Vector inner product:
![SVM](\images\SVM_inner_product.jfif)

### Explainaton of margin in SVM:

![SVM](\images\SVM_margin.jfif)

As we can see from the picture, our objective of minimizing the term $\frac{1}{2} \sum_{j=1}^{n}\theta_j^2$ and $\theta^Tx^{(i)}$ can be modified into its alternative form by applying inner product. 

![SVM](\images\SVM_margin_1.jfif)

Since we want to minimize $\frac{1}{2} \sum_{j=1}^{n}\theta_j^2$, we need a small value for $||\theta||$. However, with small values of $p^{(i)}$, we will need large values of  $||\theta||$. Therefore SVM finds solution with larger values of  $p^{(i)}$ which corresponds to the margin of decision boundary.


## ```Kernels```

When dealing with non-linear decision boundaries, we might need to use complex polynomial features which is computationaly expensive. We can use kernels which are similarity functions that simplisticaly finds how close two points are from each other. For example, a gaussian kernel is given by:

$$
    f_n = exp{(-\frac{||x-l^{(n)}||^2}{2\sigma^2})}
$$

feature $f$ marks how far $x$ is from landmark $l$ since for vectors that are close, the term $||x-l^{(n)}||^2$ will be close to 0 giving $f$ a value close to 1 and vice versa. Decreasing $\sigma^2$ makes the plot narrower which makes the feature become 0 more easily and opposite happens when we increase $\sigma^2$.

given $m$ training examples we can define our landmarks as $l^{(i)} = x^{(i)}$ such that we get $m$ number of landmarks thus we can define our new features as:

$$
    f^{(i)} = similarity(x,l^{(i)})
$$

Putting it into SVM function we get:

$$
    min_{\theta} \  C\sum_{i=1}^{m} [
        y^{(i)}cost_1(\theta^Tf^{(i)}) +
        (1 - y^{(i)})cost_0(\theta^Tf^{(i)})
        ] +
        \frac{1}{2} \sum_{i=1}^{m}\theta_j^2 \\
$$

### bias and variances:
![SVM bias vs variance](\images\SVM_sigma_bias_var.jfif)

---
## Extras on using SVM

- Use libraries
- do feature scaling before implementing kernels
- kernels implementation need to satisfy Merscer's Theorem
    - polynomial kernel
    - string kernel
    - chi-square kernel
    - hitogram intersection kernel
- multiclass can use one vs all, or some library have it implemented
- SVM is a convex optimisation problem thus always find global minima
- for reference:

![SVM logistic regression vs SVM](\images\SVM_LR_vs_SVM.jfif)