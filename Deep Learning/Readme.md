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


# vanishing gradient problem

#### sigmoid activation function 
it takes the input from the previously hidden layer and squeezes it between 0 and 1. So a value fed to the sigmoid function will always return a value between 0 and 1, no matter how big or small the value is fed.

also when finding the derivative of sigmoid activation function it will always range between 0 to 0.25

![image](https://github.com/nithinganesh1/Python/assets/122164879/6f768adf-7569-48af-80e8-d05058741d05)

The main issue related to the activation function is when the new weights and biases are calculated by the gradient descent algorithm, if these values are very small, then the updates of the weights and biases will also be very low and hence, which results in vanishing gradient problem, where the model will not learn anything.

here w new and w old are approximately equal w new ≈ w old so there is no change in weight this is called vanishing gradient problem 

```
When the input is slightly away from the coordinate origin, the gradient of the function becomes very small, almost zero. In the process of neural network
backpropagation, we all use the chain rule of differential to calculate the differential of each weight w. When the backpropagation passes through the sigmoid function, the differential on this chain is very small. Moreover, it may pass through many sigmoid functions, which will eventually cause the weight w to have little effect on the loss function, which is not conducive to the optimization of the weight. The problem is called gradient saturation or gradient dispersion.
```
- The function output is not centred on 0, which affects the efficiency of the weight update
- The sigmoid function performs exponential operations, which is slower for computers.

## Commonly used activation functions

1. Sigmoid activation function
   - smooth gradient preventing jumps in output values.
   - Output values bound between 0 and 1, normalizing the output of each neuron.
   - Clear predictions, i.e. very close to 1 or 0.
  
   > Prone to gradient vanishing
   > Function output is not zero-centered
   > Power operations are relatively time-consuming

2. tanh function
   ![image](https://github.com/nithinganesh1/Python/assets/122164879/9ea4039c-ccae-4efd-8717-7473b652669c)
    The curves of tanh function and sigmoid function are relatively similar
   - The output interval of tanh is 1), and the whole function is 0-centric, which is better than sigmod.
     ```
     tanh the output values range between −1 and 1, which is just the sigmoid function curve extended. Hence, negative inputs of the hyperbolic functions will be mapped to a negative output as well as the input values that are nearing zero will also be mapped to output values nearing zero. Therefore, the network is not stuck due to the above features during training. 
     ```
3. ReLU function
   ![image](https://github.com/nithinganesh1/Python/assets/122164879/0c743c3b-4df6-40ee-acb0-88cfa8008d32)

    f(x) =max(0,x)

   The ReLU function is actually a function that takes the maximum value. Note that this is not fully interval-derivable, but we can take sub-gradient, as shown in the figure above. Although ReLU is simple, it is an important achievement in recent years.

The ReLU (Rectified Linear Unit) function is an activation function that is currently more popular. Compared with the sigmod function and the tanh function, it has the following advantages:
 - When the input is positive, there is no gradient saturation problem.
 -  The calculation speed is much faster. The ReLU function has only a linear relationship. Whether it is forward or backward, it is much faster than sigmod and tanh. (Sigmod and tanh need to calculate the exponent, which will be slower.)

 >  When the input is negative, ReLU is completely inactive, which means that once a negative number is entered, ReLU will die. In this way, in the forward
propagation process, it is not a problem. Some areas are sensitive and some are insensitive. But in the backpropagation process, if you enter a negative number, the gradient will be completely zero, which has the same problem as the sigmod function and tanh function.
 > we find that the output of the ReLU function is either 0 or a positive number, which means that the ReLU function is not a 0-centric function.

4. Leaky ReLU function
   To solve the Dead ReLU Problem, people proposed to set the first half of ReLU 0.01x instead of 0.
   f(x) = max(0.01,x)
   Another intuitive idea is a parameter-based method, Parametric ReLU : f(x)=  max(alpha x,x), which alpha can be learned from back propagation. In theory, Leaky ReLU has all  the advantages of ReLU, plus there will be no problems with Dead ReLU, but in actual operation,  it has not been fully proved that Leaky ReLU is always better than ReLU.
   
   - if a=0, f becomes ReLU
   - if a>0, f becomes leaky ReLU
   - if ai is a learnable parameter, f becomes PReLU

5. Softmax
   The softmax activation function is used to turn a vector of numbers into a vector of probabilities that add up to one. It does this by making the larger numbers in the vector more likely and the smaller numbers less likely. 
#### Activation function we should use.
                          | Hidden Layer | Output Layer |
  _______________________________________________________
  - Binary classification | Relu/Relu Variations.|   sigmoid    |
  - Multiy class_________| Relu/Relu Variations.|  softmax     |
  - Regression__________| Relu/Relu Variations.| Linear       |




