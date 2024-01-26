Deep Learning

AI vs ML vs DL vs Data Science:
	AI: Applications that can perform tasks without human intervention. Examples: Netflix recommendations, self-driving cars.
	ML: A subset of AI that uses statistical tools to analyze data and make predictions.
	DL: A subset of ML that uses artificial neural networks to learn from data.
	Data Science: A field that combines ML, statistics, and domain knowledge to solve problems.

Why Deep learning is becoming popular?	
	Availability of Big Data: The amount of data that is being generated is increasing exponentially. Eg. Facebook, Tweeter, LinkedIn made huge amount of data
	Advancements in Hardware: The development of powerful GPUs has made it possible to train deep learning models much faster than before. This has made deep learning more accessible to a wider range of people.
	Improved Algorithms: Deep learning algorithms are constantly being improved. This has led to the development of new and more powerful models that can solve a wider range of problems.
	Success in Various Applications: Deep learning has been successful in a variety of applications, such as image recognition, natural language processing, and self-driving cars. This has led to increased interest in deep learning from both businesses and individuals.

Perceptron
A single layer or multiple Layer of neural network

Weights: These are numerical values assigned to the connections between neurons in different layers of the network. They determine the strength of the signal from one neuron to another. Higher weights indicate a stronger influence, while lower weights represent a weaker influence

Input x1
                            W1
Input x2           W2	hidden Layer1		output layer1
                            W3
Input x3

There can be multiple hidden layers ie, multi-layer perceptrons (MLPs), also known as artificial neural networks with multiple layers. MLPs have multiple layers of neurons, with each layer connected to the next. This allows them to learn more complex relationships in the data
Y = ∑xiwi +b
Biases: These are constant values added to the output of a neuron before applying the activation function. They introduce a shift in the activation of the neuron, allowing it to respond differently to the same input signals.

Activation Function:
How much level of neuron should activate or deactivate
Eg.
Sigmoid activation function
				 = 1/1+e-y

Basically, used for binary classification functions ie, 0 or 1
Linear activation function 

Forward And Backword propagation 
Algorithms used to train artificial neural networks.
Forward Propagation: 
	Input
	Multiple by wight
	Add bias
	Activate function 
If we got wrong output ie, if predicted value y ̂ != y then we need to use loss function y-y ̂
If the Loss function is huge,Optimise the Wight and update the wights
Eg, Gradient dissent
Backward Propagation:
	Calculate loss function 
	Optimiser

