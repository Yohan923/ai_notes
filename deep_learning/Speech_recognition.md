# Speech Recognition

## Language model:

Estimates the probabilty of a sentence.

- Tokenize words in the sentence. \<EOS\>(end of sentence token), \<UNK\> (unknown word)

RNN model:

- for input $x$ one output $y$ outputs the softmax for the probability of the next word in the whole dictionary
- pass on the output to the next input.

## Sampling novel sequences:

- starting with $a^{<0>}$ and $x^{<0>}$, generate an output
- randomly sample the next word from output and pass as input to the next.
- end when we get end of sentence or a number of time steps
- can generate unknow word token can either reject or just leave it

## Models:

CTC model:

- equal number of input and output
- inputs different time frames of the audio input
- outputs collapsed to produce final sentence

![**CTC_model**](images/speech_recog_CTC.jfif)

Attention model:

- inputs different time frames of the audio input
- outputs characters with attention model

![**attention model**](images/speech_recog_attention.jfif)

---

## Trigger word:

- define the point in audio clip where the person has just finished saying the trigger word. set the target label to $1$
- set $1$ to several time steps after the trigger word to slightly even out the ratio of $0$ to $1$

![**trigger word**](images/trigger_word.jfif)