# Deep Learning Overview

## AI vs ML vs DL vs Data Science:

- **AI (Artificial Intelligence):** Applications that can perform tasks without human intervention. Examples include Netflix recommendations and self-driving cars.

- **ML (Machine Learning):** A subset of AI that uses statistical tools to analyze data and make predictions.

- **DL (Deep Learning):** A subset of ML that employs artificial neural networks to learn from data.

- **Data Science:** A field that combines ML, statistics, and domain knowledge to solve problems.

## Why Deep Learning is becoming popular?

1. **Availability of Big Data:** The exponential increase in data generation from platforms like Facebook, Twitter, and LinkedIn.

2. **Advancements in Hardware:** Powerful GPUs have accelerated the training of deep learning models, making it more accessible.

3. **Improved Algorithms:** Constant enhancements in deep learning algorithms leading to the development of more powerful models.

4. **Success in Various Applications:** Deep learning has shown success in image recognition, natural language processing, and self-driving cars, sparking increased interest.

## Perceptron:

- A single layer or multiple layers of a neural network.

### Components:
 ![image](https://github.com/nithinganesh1/Python/assets/122164879/0cfe0e09-c1bf-4bcb-a497-c60b5a9f8951)
- **Weights:** Numerical values assigned to connections between neurons determining signal strength.

```
Y = ∑xiwi +b
```
- **Biases:** Constant values added to neuron output before applying the activation function, introducing a shift in activation.

### Multi-Layer Perceptrons (MLPs):
 ![image](https://github.com/nithinganesh1/Python/assets/122164879/828a7d0f-ee8a-4dc2-8424-f7ffd6d9c20f)


- Neural networks with multiple layers allowing for learning complex relationships in data.

### Activation Function:

- Determines the level of neuron activation.

    - **Sigmoid Activation Function:**
      ```
      f(x) = 1 / (1 + e^(-y))
      ```
      Used for binary classification (0 or 1).

    - **Linear Activation Function:**
      Used for other types of problems.

### Forward and Backward Propagation:

- Algorithms used to train artificial neural networks.

#### Forward Propagation:

1. Input
2. Multiply by weight
3. Add bias
4. Activate function
5. Calculate loss if the predicted value is incorrect
6. Optimize weights using algorithms like gradient descent

## Backward Propagation:

1. Calculate loss function
2. Optimize using an optimizer
3. update wight

a.wight updation formula

 loss = (y-y^hat)
 to reduce loss we need to update wights
 ```
 w new = w old - learning rate 
 ```
 (∂L/∂w old) = slope 
![image](https://github.com/nithinganesh1/Python/assets/122164879/c34f36ad-1035-4499-90d3-905fa32daba3)

 We want to minimize the function by updating the weights with a learning rate.
 If the learning rate is too high, the model may overshoot the minimum and become unstable. If the learning rate is too low, the model may take too long to converge or get stuck at a suboptimal point. A common recommendation for the initial learning rate is 0.001
![image](https://github.com/nithinganesh1/Python/assets/122164879/cf1221f2-6e65-4e6f-8793-13e422e68bb0)

b. chain rule in differentiation

The chain rule is a formula used in differential calculus to calculate the derivative of a composite function.
![image](https://github.com/nithinganesh1/Python/assets/122164879/07ed8b18-8a01-417e-a05e-4eef714b48de)

#### updating w4
```
∂L/∂w4 new = ∂L/∂O2 * ∂O2/∂w4
```
O2 and w4 are related 
we can cancel O2 when cross-multiplication 

####updating w1
```
∂L/∂w1 new = ∂L/∂O2 * ∂O2/∂O1 * ∂O1/∂w1
```

![image](https://github.com/nithinganesh1/Python/assets/122164879/1ff6a1f7-4dfd-4ea5-be04-5da8eb54f6e6)


#### updating w1
```
∂L/∂W1 new = [ ∂L/∂31 * ∂31/∂21 * ∂21/∂11 * ∂11/w1 ] + [ ∂l/∂31 * ∂31/∂22 * ∂22/∂11 * ∂11/∂w1 ]
```
need to calculate the total value of all the new w1 that I am receiving from different ways. when there multiple Hidden layers



