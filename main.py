import math
import torch 
import torch.nn as nn 
from torch.nn import functional as F


# Hyperparameters
# -----------------------------------------------------
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
d_model = 512 # hidden representation for what a token is 
num_heads = 8 # number of heads in multi headed attention
d_k = d_model // num_heads # hidden layer for attention 
learning_rate = 3e-4
max_iterations = 100000
eval_iters = 200
n_layers = 6
dropout = 0.1
# -----------------------------------------------------

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    return x.to(device), y.to(device)
# -----------------------------------------------------

# Define the model 
# -----------------------------------------------------
class FeedForward(nn.Module): 
    def __init__(self, d_model):
        super().__init__() 
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, inputs):
        return self.net(inputs)

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Embedding(block_size, d_model)

    def forward(self, inputs):
        B, T = inputs.shape

        embeddings = self.embedding(inputs) * math.sqrt(d_model) # (B, T, d_model)
        pos_encoding = self.positional_encoding(torch.arange(T, device=inputs.device)) # (T, d_model)

        return embeddings + pos_encoding # (B, T, d_model)

class CausalAttention(nn.Module):
    def __init__(self, d_model, d_k): 
        super().__init__()
        self.query = nn.Linear(d_model, d_k) 
        self.key = nn.Linear(d_model, d_k) 
        self.value = nn.Linear(d_model, d_k) 
        self.dropout = nn.Dropout(dropout)

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
        attention = self.dropout(attention)
        output = attention @ v  

        return output # (B, T, d_k)

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_model=d_model, d_k=d_model//num_heads) for _ in range(num_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        out = torch.cat([h(inputs) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multi_headed_attention = MultiHeadedAttention(num_heads=num_heads, d_model=d_model)
        self.feed_forward = FeedForward(d_model=d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.multi_headed_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads):
        super().__init__()
        self.embedding = Embedding(vocab_size=vocab_size, d_model=d_model) # -> (B, T, d_model)
        self.blocks = nn.Sequential(*[Block(d_model=d_model, num_heads=num_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size) # -> (B, T, vocab_size) for logits

    def forward(self, inputs, targets=None):
        embeddings = self.embedding(inputs)
        scores = self.blocks(embeddings)
        scores = self.ln(scores)
        logits = self.linear(scores)

        if targets == None: 
            return logits, None
        
        B, T, C = logits.shape 

        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, inputs, num_tokens):
        # inputs are (B, T)
        for i in range(num_tokens):
          # Crop idx to last block size
          inputs_cond = inputs[:, -block_size:]
          logits, _ = self(inputs_cond)
          # Get last token for each row
          logits = logits[:, -1, :] # (B, C)
          # Softmax it and sample the highest probability token
          probs = F.softmax(logits, dim=-1)
          idx_next = torch.multinomial(probs, num_samples=1)
          # append that token to inputs
          inputs = torch.cat((inputs, idx_next), dim=1)
        return inputs 

# -----------------------------------------------------

# Training the model 
# -----------------------------------------------------


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = Decoder(vocab_size=vocab_size, d_model=d_model, num_heads=num_heads).to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_log = []
for step in range(max_iterations):
    xb, yb = get_batch('train')

    if step % 500 == 0 or step == max_iterations - 1:
        losses = estimate_loss()
        log_line = f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        print(log_line)
        loss_log.append(log_line)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

with open('loss_log.txt', 'w', encoding='utf-8') as f:
    for line in loss_log:
        f.write(line + '\n')
# -----------------------------------------------------

# Sample generation
sample_input = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(sample_input, num_tokens=10000)
generated_text = decode(generated[0].tolist())
print(generated_text)
with open('generated.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)
