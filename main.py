import math
import torch 
import torch.nn as nn 
from torch.nn import functional as F


# Hyperparameters
# -----------------------------------------------------
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
d_model = 32 # hidden representation for what a token is 
d_k = 64 # hidden layer for attention 
# -----------------------------------------------------

# Process and tokenize the input data
# -----------------------------------------------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# split data into test and train
len_data = len(text)
num_train = int(len_data * 0.9)

train_data = text[:num_train]
test_data = text[num_train:]

# tokenize data
tokens = sorted(list(set(text)))
vocab_size = len(tokens)

stoi = { ch:i for i,ch in enumerate(tokens) }
itos = { i:ch for i,ch in enumerate(tokens) }
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[i] for i in x])

# Encode the data as tensors of token indices
train_data = torch.tensor(encode(train_data), dtype=torch.long)
test_data = torch.tensor(encode(test_data), dtype=torch.long)

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
# -----------------------------------------------------

# Define the model 
# -----------------------------------------------------
class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(block_size, d_model)

    def forward(self, inputs):
        B, T = inputs.shape

        embeddings = self.embedding(inputs) * math.sqrt(self.d_model) # (B, T, d_model)
        pos_encoding = self.positional_encoding(torch.arange(T)) # (T, d_model)

        return embeddings + pos_encoding # (batch_size, block_size, d_model)

class CausalAttention(nn.Module):
    def __init__(self): 
        super().__init__()
        self.query = nn.Linear(d_model, d_k) 
        self.key = nn.Linear(d_model, d_k) 
        self.value = nn.Linear(d_model, d_k) 

    def forward(self, inputs):
        B, T, _ = inputs.shape

        q = self.query(inputs) # (B, T, d_k)
        k = self.key(inputs) # (B, T, d_k)
        v = self.value(inputs) # (B, T, d_k)

        # Compute attention scores
        scores = q @ torch.transpose(k, -2, -1) # (B, T, T)

        # Create causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=inputs.device)).unsqueeze(0)  # (1, T, T)
        scores = scores.masked_fill(causal_mask == 0, -1e9)  # Apply mask

        # Scale and compute attention
        attention = torch.softmax(scores / math.sqrt(d_k), dim=-1)  # (B, T, T)
        output = attention @ v  

        return output # (B, T, d_k)


# -----------------------------------------------------

# Training the model 
# -----------------------------------------------------

# -----------------------------------------------------
