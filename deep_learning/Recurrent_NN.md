# Recurrent Neural Network

Used for sequence data:

- speech recognition
- music generation
- sentiment classification
- DNA sequence analysis
- machine translation
- video activity recognition
- name entity recognition

Types:

![**RNN types**](images/RNN_types.jfif)

Notations:

- $T_x, T_y$ - number of input and output sequence
- 

Forward prop:

$$
    \begin{aligned}
    W_a[a^{t-1},x^{<t>}] &= W_{aa}a^{t-1} + W_{ax}x^{<t>} \\
    a^{<t>} &= g(W_a[a^{t-1},x^{<t>}] + b_a) \\
    \hat{y} &= g(W_{ya}a^{<t>} + b_y)
    \end{aligned}
$$

- $W_a$ is obtained by concatenating matrices $W_{aa}$ and $W_{ax}$
- $[a^{t-1},x^{<t>}]$ is obtained by stacking $a^{<t-1>}$ on top of $x^{<t>}$

Lost/Cost:

$$
    \begin{aligned}
    L^{<t>}(\hat{y}^{<t>},y^{<t>}) &= - y^{<t>}log(\hat{y}^{<t>}) - (1-y^{<t>})log(1-\hat{y}^{<t>}) \\
    L(\hat{y}, y) &= \sum_{t=1}^{T_y} L^{<t>}(\hat{y}^{<t>},y^{<t>}) \\
    \end{aligned}
$$

Back prop through time: 

- use cost to calculate Back prop backwards

---

## Vanishing gradients:

words deeper in the sentence does not get effected from words earlier eg. (cat,was) / (cats,were)

- exploding gradient are not very serious problem, usually numerical overflow. (can clip gradients, clip gradient if some threshold is reached)

## Gated Recurrent Unit(GRU):

- $c^{<t>}$ = memory cell
- $\tilde{c}^{<t>}$ - candidate for $c$.
- $\Gamma_u$ - update gate controls whether to update $c$ with $\tilde{c}^{<t>}$ 
- $\Gamma_r$ - relevance, how relevant is $c^{<t-1>}$ to calculate $c^{<t>}$

$$
    \begin{aligned}
    \tilde{c}^{<t>} &= tanh(W_c[\Gamma_r * c^{<t-1>}, x^{<t>}] + b_c) \\
    \Gamma_u &= \sigma(W_u[c^{<t-1>}, x^{<t>}] + b_u) \\
    \Gamma_r &= \sigma(W_r[c^{<t-1>}, x^{<t>}] + b_r) \\
    c^{<t>} &= \Gamma_u * \tilde{c}^{<t>} + (1-\Gamma_u)c^{<t-1>} \\
    a^{<t>} &= c^{<t>} \\
    \end{aligned}
$$

## Long Short Term Memory (LSTM):

- $\Gamma_f$ - forget gate
- $\Gamma_o$ - output gate

$$
    \begin{aligned}
    \tilde{c}^{<t>} &= tanh(W_c[a^{<t-1>}, x^{<t>}] + b_c) \\
    \Gamma_u &= \sigma(W_u[a^{<t-1>}, x^{<t>}] + b_u) \\
    \Gamma_f &= \sigma(W_f[a^{<t-1>}, x^{<t>}] + b_f) \\
    \Gamma_o &= \sigma(W_o[a^{<t-1>}, x^{<t>}] + b_o) \\
    c^{<t>} &= \Gamma_u * \tilde{c}^{<t>} + \Gamma_f*c^{<t-1>} \\
    a^{<t>} &= \Gamma_o * tanh(c^{<t>}) \\
    \end{aligned}
$$

---

## Bidirectional RNN:

look at information from past and future

![**BRNN**](images/Bidirectional_RNN.jfif)

## Deep RNN:

- you can have many deep, non-recurrent layers after recurrent layers to compute $y$
- since current layers are expensive, you won't see many deep recurrent layers, only three recurrent layers are common.
- each unit in the layers can be normal RNN, DRU, LSTM or BRNN

![**Deep RNN**](images/Deep_RNN.jfif)