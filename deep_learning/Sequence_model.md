# Sequence model

## Basic model:

- input a sequence, then output sequence. pretty much has encoder part of many to many RNN where $T_x \ne T_y$ then the decoder part is similar to a one to many RNN except that you are only given $a^{<0>}$ from encoder to calculate $y^{<1>}$

## Machine Translation:

- trying to find the most likely translation:

$$
    arg \  max \  P(y^{<1>},...,y^{<T_y>} | x)
$$


### ```Beam search```:

- $B$ - beam width, the number of words to consider.

1. consider $B$ most likely words.
2. for each considered word, use a seperate model for each word to predict the next word. 
3. the again choose the $B$ most likely combinations

![**Beam Search**](images/Beam_search.jfif)

Length normalization:

- since taking product of probabilities can lead to underflow, we take sum of log of the probabilities.
- product of probabilities also tend to favour short sentences.

![**Length Norm**](images/Length_norm.jfif)

- $\frac{1}{T_y^{\alpha}}$ is a normalizing constant, $\alpha = 1$ means full normalize $\alpha = 0$ means no normalize. can be tunned to get best result
- for each of the results you get from beam search you calculate the log objective, and gets the highest one as the final result
- large $B$ then you might get better result, but require more memory and lower. smaller $B$ is faster but less good
- Beam search algorithm does not guarentee to find the exact maximum.

Error analysis:

- $y^*$ - Human translation
- $\hat{y}$ - machine translation

1. if $P(y^*|x) > P(\hat{y}|x)$, beam search gets $\hat{y}$ but human translation is higher than beam search is at fault
2. if $P(y^*|x) \le P(\hat{y}|x)$, $y^*$ is better than $\hat{y}$, but RNN predict otherwise, thus RNN might have some problems.

```Bleu Score```:

- single real number evaluation of how good a translation is

### ```Attention Model```:

- no need to remember long chunk of sentences, we can only look at a section of the entire sentence for translation like human does.

- generate attention weights $\alpha$ which tells us how much attention we should be paying to each of the words when translation a word.

![**attention model**](images/attention_model.jfif)

![**Attention weights**](images/attention_weights.jfif)
