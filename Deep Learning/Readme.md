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
 w new = w old - learning rate * slop
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
   - The output interval of tanh is 1), and the whole function is 0-centric, which is better than sigmoid.
     ```
     tanh the output values range between −1 and 1, which is just the sigmoid function curve extended. Hence, negative inputs of the hyperbolic functions will be mapped to a negative output as well as the input values that are nearing zero will also be mapped to output values nearing zero. Therefore, the network is not stuck due to the above features during training. 
     ```
3. ReLU function
   ![image](https://github.com/nithinganesh1/Python/assets/122164879/0c743c3b-4df6-40ee-acb0-88cfa8008d32)

    f(x) =max(0,x)

   The ReLU function is actually a function that takes the maximum value. Note that this is not fully interval-derivable, but we can take a sub-gradient, as shown in the figure above. Although ReLU is simple, it is an important achievement in recent years.

The ReLU (Rectified Linear Unit) function is an activation function that is currently more popular. Compared with the sigmoid function and the tanh function, it has the following advantages:
 - When the input is positive, there is no gradient saturation problem.
 -  The calculation speed is much faster. The ReLU function has only a linear relationship. Whether it is forward or backwards, it is much faster than sigmod and tanh. (Sigmod and Tanh need to calculate the exponent, which will be slower.)

 >  When the input is negative, ReLU is completely inactive, which means that once a negative number is entered, ReLU will die. In this way, in the forward
propagation process, it is not a problem. Some areas are sensitive and some are insensitive. But in the backpropagation process, if you enter a negative number, the gradient will be completely zero, which has the same problem as the sigmoid function and tanh function.
 > We find that the output of the ReLU function is either 0 or a positive number, which means that the ReLU function is not a 0-centric function.

4. Leaky ReLU function
   To solve the Dead ReLU Problem, people proposed to set the first half of ReLU 0.01x instead of 0.
   f(x) = max(0.01,x)
   Another intuitive idea is a parameter-based method, Parametric ReLU: f(x)=  max(alpha x,x), in which alpha can be learned from backpropagation. In theory, Leaky ReLU has all  the advantages of ReLU, plus there will be no problems with Dead ReLU. Still, in actual operation,  it has not been fully proved that Leaky ReLU is always better than ReLU.
   
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


## optimizers
### Gradient Descent

![image](https://github.com/nithinganesh1/Python/assets/122164879/b07e1313-4367-4cf0-acf9-6333367553c3)

for updating the weight we need to minimise the loss function so we use optimisers an example of an optimiser is gradient descent.

#### EPOCH
```
1 EPOCH = one forward and backwards propagation
```
Disadvantage of Gradient Descent
1. Convergence is very slow
2. Resource Extensive (it needs to have a huge RAM in our system)

### Stochastic gradient descent

unlike the Gradient Descent, it runs only one Epoch at a time.
1. need not that much RAM 
2. also it will decrease the Speed 

### Mini Batch SGD

1. it uses a small subset (mini-batch) of training samples.
2. Each mini-batch contributes to a single iteration of the optimization algorithm.
3. Resource not too much
4. speed is better than stochastic
5. convergence is better
6. time complexity will improve 

![image](https://github.com/nithinganesh1/Python/assets/122164879/c4e66d0d-c714-4eb9-b7b8-a9716da0cf75)

### SGD with Momentum 

In all that above SGD has the noise we need to remove the noise so we use Momentum  We need to make sure we smoothen the entire process of reaching the global minima.

#### Exponential moving average

```
 w new = w old - learning rate *  (∂L/∂w old)
```
here we update the weight updation formula,
```
wt = wt-1 - learning rate *  (∂L/∂w t-1)
```
1. wt = current weight
2. wt-1 = old weight

t1, t2, t3, t4,...tn
a1, a2, a3, a4,...an

1. t = time
2. a = value of corresponding time
3. v = value of

vt1 = a1
vt2 = beta * vt1 + (1-beta) * a2

beta value is always between 0 to 1 if the beta value is .95 then it is more important to my vt1 if the previous time stamp value is near zero then we give more importance to the current timestamp value (a2).

vt3 = beata * vt2 + (1-beta) * a3

```
wt = w(t-1) - learingrate * Vdw
```
Vdw = value of derivative ie, beta* Vdw(t-1) + (1-beta) *  (∂L/∂w t-1)
This will smoothen the corve

1. Reducing the noise
2. Mini batch
3. Quicker convergence

### Adagrad --> Adaptive Gradient Descent

is all the previous Gradient Descent the learning rate is constant when we reach the global minima also it should decrease the learning rate it will be more logical 

```
wt = wt-1 - learning rate *  (∂L/∂w t-1)
```
we will change the learning rate to eta

```
wt = wt-1 - learning rate eta*  (∂L/∂w t-1)
```
learning rate eta = 
```
learning rate/sqrt(alfa t + E)
```
E = epsilon = the denominator never becomes 0
```
alpha t = ∑ n i = 1 ((∂L/∂w t)^2
```

 ∑ n i = 1 ((∂L/∂w t)^2 Here we using summation so the alfa of t will always increase when the t will increase
 We are dividing the learning rate by the increasing alpha (learning rate eta = learning rate/sqrt(alfa t + E) ) so the learning rate will decrease every t will increase

 ### Adadelta and rmsprop

if we keep updating alpha then we divide the learning rate it is already a small number when replacing the learning rate is that very small number wt1 wont change that much ie
wt approximately equal to wt-1

so we need to do alpha of t never so big so we change the alpha 
```
learning rate eta = learing rate/sqrt(sdw + E)
```
using Exponential weighted avg
Sdw = beta sdw(t-1) + (1-beta)(∂L/∂w t-1)^2

like this we control using Exponentially weighted avg this value will never increase very huge value sing the beta value because there is only a slow decrease in the learning rate also we will reach global minima.

### Adam Optimizer

comparing momentum + PMSPROP for smoothening and decreasing noise.

Momentem + RMSPROP

### Feature scaling 
Eg.
ANN ✔️
Lr ✔️
DT/RF ❌
XGB ❌
KNN ✔️
Kmeans ✔️
1. anything that is distance based, ie, values are bigger make time
2. if there is gradient decent included must need

### Tensorflow 
Opencored by Google, Pytorch by Facebook
TensorFlow v>2.0 has Keras integrated 

## Notes

1. Sequential
   * the entire layers input layers, hidden layers, and output layers are interconnected theses all the nurl-network together ie, the block is called a sequential
   * ie, a huge block that has a neural network inside it, so it can differently do forward and backwards propagation.
2. Dense layer
   * It will help to create layers, it will create input layers, hidden layers and output layers
3. Activation functions
   * for using the activation function we will able to use it
4. Dropout layer
   * Sometimes the entire neural network leads to overfitting
   * cut out some of the connections that will deactivated

### black box model vs white box model
Random forest = Black box model
DT = White box 

Black box: a magic machine that gives answers, but you can't see how it works inside (you don't know why it says what it says)

White box: you know exactly what ingredients go in and how they're mixed, so you understand why it makes a certain dish. Clear as cooking!

# Convolutional Neural Network (CNN)
## CNN VS Humen Brain
Our brains are the master at determining objects quickly.
we can differentiate cats and dogs and if anyone dressed like cats we can differentiate them easily like this picture 
![image](https://github.com/nithinganesh1/Python/assets/122164879/b853dd49-a777-4243-88ca-03c019981afa)

in major part of our back side of the brain is called the Cerebral Cortex there is a part of the brain called the Visual cortex which is specifically responsible  for seeing any items that are there any object that are there in what we see. In visual cortex has many layers.

![Presentation1](https://github.com/nithinganesh1/Python/assets/122164879/eb00cad1-62d7-4036-b9ed-3a999801025f)

Each layer will do some functionality, ie, something is happening in every layer and finally, all the information is passed then we can visualise the v7 layer image ie, and we are finally finding the output after so much processing

Also CNN should be also able to do this many numbers of different kinds of processing

### 1. Convolution
Basic information about images
1. it has pixels
2. 2 types of photos black and wight and RGB type
### Black and wight
1. Consists of only two colours
2. Every pixel ranges between 0 to 255: Black (0) and White (255).

 ### 2. RGB image (coloured image)
1. Represented using three colour channels: Red, Green, and Blue.
2. 3 channels R, G, B channels
3. ![Presentation3](https://github.com/nithinganesh1/Python/assets/122164879/fdbb8bf5-a3b7-4b59-a14e-6e35c3ba988b)
4. this is the representation of RGB in pixel format 5*5*3 channel
5. every channel value ranges between 0 to 255 when this campaign will get colour.

Convolution operation.

   


