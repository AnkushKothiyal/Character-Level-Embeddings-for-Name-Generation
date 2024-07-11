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
from torch.nn import Dropout

g = torch.Generator().manual_seed(2147483647)

with open('names.txt', 'r') as f:
    words = f.read().splitlines()

# build a vocublary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {value:key for key,value in stoi.items()}

vocab_size = len(itos)
block_size = 3  # the context length i.e. how many characters does the NN takes into account to predict the next character/token
def build_dataset(words):
  X, Y = [], []
  for w in words:

    #print(w)
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# Hyperparameters
lookup_dim = 8 # The size of the word embedding Matrix i.e. the dimension of the vectors to represent in each character/token
neurons_hidden_layer = 110

C = torch.randn(vocab_size,lookup_dim, generator=g)
W1 = torch.randn((block_size*lookup_dim, neurons_hidden_layer), generator=g)
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

loss_array = []
lr = 0.2
n = 100000
for i in range(n):

    #minibatch construct
    batch_size = random.randint(100, 200)
    ix = torch.randint(0, Xtr.shape[0],(batch_size,))

    ## forward pass
    emb = C[Xtr[ix]]     # Layer 1: The lookup matrix
    h = torch.tanh(emb.view(-1, lookup_dim*block_size) @ W1 + b1)    # Layer 2: The hidden layer
    h2 = torch.tanh(h @ W2 + b2)     # Layer 3: Second hidden layer
    logits = h2 @ W3 + b3     # Layer 3: the output layer

    loss = F.cross_entropy(logits, Ytr[ix])     # loss function
    loss_array.append(loss.item())


    if i%(n/10) == 0:
        if i!=0: lr = round(lr/1.1,3)
        print(f"loss at step {i} = ",loss.item())
        print(f"Learning rate at step {i} = ",lr)

    ## backward pass
    for p in NN_parameters:
        p.grad = None
    loss.backward() #backward propogation
    for p in NN_parameters:
        p.data += -lr * p.grad #updating


# training loss 
emb = C[Xtr]     # Layer 1: The lookup matrix
h = torch.tanh(emb.view(-1, lookup_dim*block_size) @ W1 + b1)     # Layer 2: The hidden layer
h2 = torch.tanh(h @ W2 + b2)
logits = h2 @ W3 + b3  
training_loss = F.cross_entropy(logits, Ytr)
print("loss on train dataset = ", training_loss.item())

# loss on train dataset
emb = C[Xdev]     # Layer 1: The lookup matrix
h = torch.tanh(emb.view(-1, lookup_dim*block_size) @ W1 + b1)     # Layer 2: The hidden layer
h2 = torch.tanh(h @ W2 + b2)
logits = h2 @ W3 + b3  
dev_loss = F.cross_entropy(logits, Ydev)
print("loss on dev dataset = ", dev_loss.item())


torch.save(NN_parameters, 'nn_parameters_dev_216.pth')
# loss on test dataset
emb = C[Xte]     # Layer 1: The lookup matrix
h = torch.tanh(emb.view(-1, lookup_dim*block_size) @ W1 + b1)     # Layer 2: The hidden layer
h2 = torch.tanh(h @ W2 + b2)
logits = h2 @ W3 + b3  
test_loss = F.cross_entropy(logits, Yte)
print("loss on test dataset = ", test_loss.item())


# visualize dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
plt.grid('minor')