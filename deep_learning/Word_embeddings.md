# Word Embeddings

Featurized representation of words, associates similar words together using features. eg. man is to woman, king is to queen

- can estimate word analogies by calculating the similarity between pair of words
- $e_j$ - represents the embedding vector of word j, that is the column in the embedding matrix of that word.

![**Feature vector**](images/Feature_vector.jfif)

- can use t-SNE to visualize similar groups of things.

---

1. learn word embeddings from large word corps(use pretrained)
2. Transfer embedding to new task with smaller training set.(transfer learning)
3. (optinonal) Fine tunning the word embedding with new data if you have a lot of data

---

## Learing word embedding:

intuition:

generate embedding vectors from one hot vectors of words from the sentence and feed them into a NN which has softmax output to find a target word. eg. i like orage _juice_ where context can be i like orange and target is juice.

- given context words, estimate target word.
- context word can be adjusted to get best results. eg. last four words from the target word is good to build a language model
- 4 words left and right of target, nearby one word, last 1 word are good for learning word embedding.

### Skip-gram:

- pick a context word, predict whats the target word a little bit before or after the target word
- get the embedding vector for the context word and feed it to a softmax function to predict $\hat{y}$ which represents $p(t|c)$
- calculate loss using output.
- computational speed is a problem. since have to sum up all of words in dictionary. Can use hierachial softmax to help.

![**Skip gram**](images/Skip_gram.jfif)

- sampling context word randomly usually result in choosing common words more frequently such as the, and , or, etc. try use heuristics to sample less common words too.

### Negative sampling:

- more efficient than skip-gram

1. pick positive example.
2. pick $k$ negative examples with the same context word by randomly choose words from dictionary

![**Negative sample**](images/Negative_sample.jfif)

Model:

- only train $k+1$ logistic units each time therefore much faster that training 10000 softmax.

![**Negative sample model**](images/Neg_sample_model.jfif)

### GloVe(global vectors):

- $X_{ij}$ - the number of times j(target) appears in context of i(context)

Model:

![**Glove**](images/Glove.jfif)

---

## Sentiment classification:

![**Sentiment classification**](images/Sentiment_model.jfif)

## Bias in embedding:

1. identify bias direction eg. for gender, take average of $e_{male} - e_{female}$, $e_{he} - e_{she}$ etc.

2. neutralise: for every word that is not definitional, project them onto non-bias axis to get rid of bias.

3. Equalize pairs. make sure have same distance from non-bias word. eg. grandfather and grandmother are same distance from babysitter.

- binary classifier can tell you which words to neutralise
- hand pick which words to equalize