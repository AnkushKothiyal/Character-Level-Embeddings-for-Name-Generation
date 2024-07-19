# Character-Level Word Embedding Implementation
This implementation showcases a simple neural network for character-level word embedding. The model aims to represent each character in a word as a vector of fixed dimensions. These character vectors are then processed through a series of neural network layers to predict the next character in a sequence.
The dataset of a corpus of 32k names, it is divided into train, dev and test.

## Hyperparameters
- lookup_dim: The size of the word embedding matrix, representing the dimension of the vectors for each character/token.
- neurons_hidden_layer: The number of neurons in the hidden layers of the neural network.
- vocab_size: The total number of unique characters in the vocabulary.
- context_length: The size of the context window used for character predictions.

## Model Architecture
- Embedding Layer: A lookup matrix C is initialized to map each character to a vector of size lookup_dim.
- First Hidden Layer: A linear transformation followed by a non-linear activation function (tanh) is applied. Weights W1 and bias b1 are learned parameters.
- Second Hidden Layer: Similar to the first hidden layer, using weights W2 and bias b2.
- Output Layer: Produces logits for the prediction of the next character using weights W3 and bias b3.

## Training Process
- Forward Pass: The input character sequences are embedded, then passed through the hidden layers, and finally through the output layer to generate logits.
- Loss Calculation: The cross-entropy loss function is used to measure the prediction error.
- Backward Pass: Gradients are computed using backpropagation, and the model parameters are updated using gradient descent.

## Training Loop
- Initialization: Randomly initialize the model parameters.
- Mini-Batch Construction: Randomly select a batch of training examples.
- Forward and Backward Passes: Compute the loss and update the model parameters.
- Learning Rate Decay: Gradually reduce the learning rate to improve training stability.


## Performance Evaluation
- Training Loss: Calculate the loss on the training dataset after training.
- Development Loss: Evaluate the model on a separate development dataset to check for overfitting.
- Test Loss: Finally, assess the model's performance on a test dataset to estimate its generalization capability.
