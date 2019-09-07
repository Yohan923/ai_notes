# Tuning

Hyperparameter tier:

Tier 1:
- $\alpha$

Tier 2:
- $\beta$
- number of hidden units
- mini batch size

Tier 3:
- number of layers
- learning rate decay

#### Sampling:

1. Sample randomly the hyperparameters.
2. Coarse to fine

Make sure you sample to the right scale. (try log scale.)

## ```Batch Normalisation```:

Apply normalisation on the hidden units allow hidden units to also have standardised mean and varience.

Has small regularization effect.

![**Batch normalisation**](images\Batch_norm.jfif)

$\beta$ and $\gamma$ can be updated by gradient descent, RMSprop, Adam etc.

At test time:
Use exponential weighted average across mini-batches of a layer to calculate $\mu$ and $\sigma^2$


