"""
Script @author: Ankush
Related research paper: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
"""

"""
Context: The problem with n-gram models is that as we increase n, the lookup matrix becomes sparse (curse of dimensionality)
For e.g. If we switch from a character level bigram model (say only the 26 letters are considered) to a trigram character model,
 the number of combinations jump from (26 x 26) to (26 x 26 x 26), which is a big jump and thus requires significantly more data to be trained on.

In this script we'll discuss the paper which intoduces the concept of 'word embedding' i.e. we start with a neural network that assigns a vector to each
unique word/character/token and during the training process we hope that the representation evolve so that similar words end up having similar representations.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import MLP_helper

g = torch.Generator().manual_seed(2147483647)
random.seed(42)

with open('names.txt', 'r') as f:
    words = f.read().splitlines()

# build a vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {value:key for key,value in stoi.items()}

# Create train, dev and test dataset based on the context length
vocab_size = len(itos)
context_length = 3  # the context length i.e. how many characters does the NN takes into account to predict the next character/token
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr, Ytr = MLP_helper.build_dataset(words[:n1], stoi, context_length)
Xdev, Ydev = MLP_helper.build_dataset(words[n1:n2], stoi, context_length)
Xte, Yte = MLP_helper.build_dataset(words[n2:], stoi, context_length)

# Hyperparameters
lookup_dim = 8 # The size of the word embedding Matrix i.e. the dimension of the vectors to represent in each character/token
neurons_hidden_layer = 110

C = torch.randn(vocab_size,lookup_dim, generator=g)
W1 = torch.randn((context_length*lookup_dim, neurons_hidden_layer), generator=g)
b1 = torch.randn(neurons_hidden_layer, generator=g)

W2 = torch.randn((neurons_hidden_layer, neurons_hidden_layer), generator=g)
b2 = torch.randn(neurons_hidden_layer, generator=g)

W3 = torch.randn((neurons_hidden_layer, vocab_size), generator=g)
b3 = torch.randn(vocab_size, generator=g) 

NN_parameters = [C, W1, b1, W2, b2, W3, b3]

for p in NN_parameters:
    p.requires_grad = True

total_params = sum(p.nelement()  for p in NN_parameters)
print("Total parameters in the NN are: ", total_params)

Hyperparameters = [context_length, vocab_size, lookup_dim, neurons_hidden_layer]

# training the neural network
loss_array = []
lr = 0.2
n = 10000
for i in range(n):

    # forward pass
    loss = MLP_helper.forward_pass(Xtr, Ytr, NN_parameters, Hyperparameters)
    loss_array.append(loss.item())

    # print loss and adjust learning rate
    if i%(n/10) == 0:
        if i!=0: lr = round(lr/1.1,3)
        print(f"loss at step {i} = ",loss.item())
        print(f"Learning rate at step {i} = ",lr)

    ## backward pass
    MLP_helper.backward_pass(NN_parameters, learning_rate=lr, loss=loss)

# calculate loss on the three partitions of the data
train_loss = MLP_helper.compute_loss(Xtr, Ytr, NN_parameters, Hyperparameters, "train")
dev_loss = MLP_helper.compute_loss(Xdev, Ydev, NN_parameters, Hyperparameters, "dev")
# test_loss = MLP_helper.compute_loss(Xtr, Ytr, NN_parameters, Hyperparameters, "test")


# sample from the model
N_samples = 10
sample_names = MLP_helper.sample_from_model(N_samples, NN_parameters, Hyperparameters, itos)


