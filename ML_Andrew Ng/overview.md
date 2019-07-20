# Overview

Make machines study lmao

## Supervised learning  
Given date and knows what correct output looks like, teaches the relation of data to machine to make it do predictions on unseen cases. 

### Model Overview: 

> Given input features $x$ and output $y$, training example can be written as:  
>
> $$ (x^i, y^i) \ where \ i \in m$$
> $m$ is the number of training examples.  
> Training set is given to learning algorithms to produce hypothesis $h(x)$ used to make prediction.  
>  
> Accuracy of hypothesis is measured by the cost Function given by: 
> 
> $$ J(\theta) $$
> Where theta is the arugments/weight of each feature.  
> Find minimum of cost function to yield optimal hypothesis

Divided into:

(a) Regression  

- [Linear Regression](Linear_regression).

(b) Classification

- [Logistic Regression](Logistic_regression).

## Unsupervised Learning

Machine derives structure to the data on it own, we don't know how the result will look like, for example grouping data into clusters.

---

## Key Algorithms

- [Gradient Decent](Gradient_descent).
- [Normal Equation](Normal_equation.md)