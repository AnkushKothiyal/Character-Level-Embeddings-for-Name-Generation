import torch
import random
import torch.nn.functional as F


def build_dataset(words: list[str], stoi, block_size) -> tuple[torch.tensor, torch.tensor]:
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

def forward_pass(Xtr: torch.Tensor,Ytr: torch.Tensor, NN_parameters: list[torch.Tensor], Hyperparameters: list[int]) -> torch.Tensor:
    C, W1, b1, W2, b2, W3, b3 = NN_parameters
    context_length, vocab_size, lookup_dim, neurons_hidden_layer = Hyperparameters

    #minibatch construct
    batch_size = random.randint(100, 200)
    ix = torch.randint(0, Xtr.shape[0],(batch_size,))

    emb = C[Xtr[ix]]     # Layer 1: The lookup matrix
    h = torch.tanh(emb.view(-1, lookup_dim*context_length) @ W1 + b1)    # Layer 2: The hidden layer
    h2 = torch.tanh(h @ W2 + b2)     # Layer 3: Second hidden layer
    logits = h2 @ W3 + b3     # Layer 3: the output layer

    loss = F.cross_entropy(logits, Ytr[ix])     # loss function
    return loss


def backward_pass(NN_parameters: list[torch.tensor], learning_rate: float, loss: torch.Tensor):
    for p in NN_parameters:
        p.grad = None
    loss.backward() #backward propogation
    for p in NN_parameters:
        p.data += -learning_rate * p.grad #updating


def compute_loss(X_, Y_, NN_parameters: list[torch.Tensor], Hyperparameters: list[int], dataset_class: str) -> float:
    C, W1, b1, W2, b2, W3, b3 = NN_parameters
    context_length, vocab_size, lookup_dim, neurons_hidden_layer = Hyperparameters

    emb = C[X_]     # Layer 1: The lookup matrix
    h = torch.tanh(emb.view(-1, lookup_dim*context_length) @ W1 + b1)     # Layer 2: The hidden layer
    h2 = torch.tanh(h @ W2 + b2)
    logits = h2 @ W3 + b3  
    loss_ = F.cross_entropy(logits, Y_)
    print(f"loss on {dataset_class} dataset = {round(loss_.item(),5)}")

    return loss_


def sample_from_model(N_samples: int, NN_parameters: list[torch.Tensor], Hyperparameters: list[int], itos: dict) -> list[str]:
    C, W1, b1, W2, b2, W3, b3 = NN_parameters
    context_length, vocab_size, lookup_dim, neurons_hidden_layer = Hyperparameters

    g2 = torch.Generator().manual_seed(2147483647 + 154)
    all_samples = []
    for _ in range(N_samples):
        
        out = []
        context = [0] * context_length # initialize with all ...

        while True:
        
            emb = C[torch.tensor([context])]    
            h = torch.tanh(emb.view(-1, lookup_dim*context_length) @ W1 + b1)
            h2 = torch.tanh(h @ W2 + b2)
            logits = h2 @ W3 + b3
            probs = F.softmax(logits, dim=1) 

            ix = torch.multinomial(probs, num_samples=1, generator=g2).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        name = ''.join(itos[i] for i in out[:len(out)-1])

        print(name)
        all_samples.append(name)
    return all_samples 
