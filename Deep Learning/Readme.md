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

- **Weights:** Numerical values assigned to connections between neurons determining signal strength.

- ![image](https://github.com/nithinganesh1/Python/assets/122164879/0cfe0e09-c1bf-4bcb-a497-c60b5a9f8951)


- **Biases:** Constant values added to neuron output before applying the activation function, introducing a shift in activation.

### Multi-Layer Perceptrons (MLPs):
![image](https://github.com/nithinganesh1/Python/assets/122164879/828a7d0f-ee8a-4dc2-8424-f7ffd6d9c20f)


- Neural networks with multiple layers allowing for learning complex relationships in data.

### Activation Function:

- Determines the level of neuron activation.

    - **Sigmoid Activation Function:**
      ```
      f(x) = 1 / (1 + e^(-x))
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
5. Calculate loss if predicted value is incorrect
6. Optimize weights using algorithms like gradient descent

#### Backward Propagation:

1. Calculate loss function
2. Optimize using an optimizer
3. update wight


