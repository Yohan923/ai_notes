# Large Scale Machine Learning

We are dealing with very large datasets eg. m = 100,000,000.

- computation becomes very expensive using our normal algorithms.
- use learning curve to see if using large dataset is actually worth it

We can work on improving gradient descent algorithm:

## ```Stochastic gradient descent```:

We carry out gradient descent step on each example individually such that each time the inner most loop finishes, we move a small step to minimum by fitting only one example. This does not converge like normal batch gradient descent. It wonders around a region close to the global minimum.

1. randomly shuffle dataset
2. Repeat for some times<br>
{<br>
loop for $i = 1,...,m${
    
    $$
        \theta_j = \theta_j - \alpha(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
    $$

    }<br>
}

### checking convergence:

1. Before updating $\theta$ with an example, we compute the $cost(\theta, (x^{(i)},y^{(i)}))$ of the example.
2. Every $x$ iterations we plot the cost we have been compting averaged over the last $x$ examples against the number of iterations.
![Stochastic](images\Stochastic.jfif)

Since stochastic does not really converge, we may slowly lower $\alpha = \frac{const1}{iterationNumber+const2}$ to make it converge but it is at extra cost of figuring out good constant values to calculate $\alpha$ with.


## ```Mini-Batch Gradient Descent```:

Use a mini-batch size $b$ in each iteration.

Algorithm:

1. Pick batch size $b$, eg. $b=10$
2. Repeat for some times<br>
{<br>
loop for $i = 1,11,21,...,(m-b+1)${
    
    $$
        \theta_j = \theta_j - \alpha \frac{1}{b} \sum^{i+b-1}_{k=i}(h_\theta(x^{(k)})-y^{(k)})x_j^{(k)}
    $$

    }<br>
}

## ```Map-reduce and data parallelism```

Split dataset into smaller subsets. Each machine uses different subsets to do the summation part of the gradient descent. Master machine will combine these summations to use in gradient descent. Multi-core operates the same. Some libraries can parallel vectorised implementation automatically

---
## Extras

### Mini-Batch vs Stochastic:

- mini-batch can be vectorized which means it can be parallelised to increase performance this might become fater than stochastic.
- but mini-batch has a new value b that we need to spend time wondering about

### Online learning:

Continuous stream of data from an online setting. We can dynamically update parameters as the users uses the website with that particular example and discard it since we have a continuous stream of data coming in.

