# Neural Network from Scratch 🧠💻

_Learn. Build. Understand. Repeat._

---

## Overview
This is my fully **custom implementation of a neural network in pure Python and NumPy** — no PyTorch, no TensorFlow, no shortcuts. It supports multiple layers, activation functions, L2 regularization, batch training, and even learning rate schedulers.  

I built this project not just to create a functioning neural network, but to **truly understand every line of code behind the scenes** — from forward and backward passes to gradient updates. Every bug I fixed and every formula I implemented taught me more about the inner workings of deep learning than I could have imagined.

---

## Features

- **Fully connected layers** with customizable number of neurons  
- **Activations supported**: ReLU, Sigmoid, Tanh, Linear  
- **Loss function**: Mean Squared Error (MSE) with backward gradient  
- **L2 Regularization** for better generalization  
- **Learning rate schedulers**: StepLR and ExponentialLR  
- **Mini-batch gradient descent** for scalable training  
- **Parameter saving and loading**  
- **Training history** with loss and learning rate tracking  

---

## Why I Built This

- To **learn deep learning fundamentals hands-on**  
- To understand **backpropagation and gradient flow** intimately  
- To explore **customizable learning rate scheduling**  
- To experience the **full process of designing a neural network from scratch**, instead of using high-level libraries  

Every feature in this network reflects a concept I learned while building it — from initializing weights to computing derivatives and updating parameters manually.

---

## Quick Example
```python
import numpy as np
from neural_network import NeuralNetwork

# Toy dataset: XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

layer_config = [
    {'n_inputs': 2, 'n_neurons': 4, 'activation': 'tanh'},
    {'n_inputs': 4, 'n_neurons': 1, 'activation': 'tanh'}
]

nn = NeuralNetwork(layer_config, learning_rate=0.1)

history = nn.fit(X, y, epochs=1000, batch_size=4, verbose=True)

predictions = nn.predict(X)
print("Predictions:\n", predictions)
```

## Learning Highlights
- Implemented forward propagation and backpropagation from scratch
- Learned how to manually compute gradients for each layer
- Understood how L2 regularization affects weight updates
- Explored learning rate decay strategies
- Gained hands-on experience with vectorized NumPy operations for efficiency
- Developed a deeper appreciation for how frameworks like PyTorch/TensorFlow work under the hood



## Conclusion
This project isn’t just code, it’s **proof of my learning journey** in deep learning fundamentals. Every line of this network represents a step in mastering neural networks at a low level, and it has given me **confidence to experiment with more complex architectures in the future**.
