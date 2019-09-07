# Strategy

- ```Do somthing quick then iterate.```

## Single real number evaluation:
![**Precision and recall**](images/Precision_recall.jfif)

F score given by: 

$$
    F_1 = 2 \frac{PR}{P+R}
$$

can be used to approximate better alogorithms, the higher the F score he better in general.

---

- set up satisficing(e.g running time) and and optimzizing metric(e.g accuracy).

- have same or atleast similar data distribution for dev and test set.

- dev and test set should reflect future data distribution.

- choose metrics to reflect the actually data that the product will be working on when shipped

---

- Bayes optimal error - best possible error

- if there is a greater gap between human level error and train error than there is bias problem(avoidable bias). Greater gap between train error and dev error, should focus more on varience.

-  carry out error analysis to check what kind of error is mainly causing trouble then work on it accordingly.

---
### Data miss match:

- Have a training-dev set, same distribution as training set but are not used for training. 

- There is a data miss match problem when there is a high gap from train-dev set to the dev set. But high varience problem when high gap between train set and train-dev set.

- do manual analysis on data miss match.

---

## Transfer Learning:

Use when data of network B << A when you want to transfer from A to B, task A and B have same data, low level feature are helpful to the transfer.

- train a first network on one task, remove the last output layer. 

- Insert the new output layer(or mulitple additional layers.) with random initialized weight of the new task.

    1. pre-train: training on first network's data to pre-initialize all the weights
    2. fine-tunning: updating all the weights, then train on second network's data

- with small data set on second network you can only re-train the last/last few layer.


---

## Multi-task

Use when tasks have similar low level features, amount of data is similar, can train a big enough network.

- train network to do multiple things, ie. have multiple output labels. Unlike softmax. Multi-task can have multiple labels with 1. 

- Not all labels need to be provided ie. they can be unknown

## End-to-End

single pipeline system to go from X to output Y leaving out intermediate steps. Such as audio -> features -> phonems -> words -> transcript.

- normaly used when there are lots of data that can be used for training otherwise non-end-to-end might be better.

