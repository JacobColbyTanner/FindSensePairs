import torch
import torch.nn as nn
import torch.optim as optim
from CTRNN import RNNNet, TransformRNNNet
import numpy as np
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Load PTB dataset
train_iter = PennTreebank(split='train')
tokenizer = get_tokenizer('basic_english')

# Build vocabulary


def yield_tokens(data_iter):
    for line in data_iter:
        yield tokenizer(line)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=[''])
vocab.set_default_index(vocab[''])

# Prepare dataset


def data_process(raw_text_iter):
    data = [torch.tensor([vocab[token] for token in tokenizer(
        item)], dtype=torch.long) for item in raw_text_iter]
    return data


train_data = data_process(PennTreebank(split='train'))
val_data = data_process(PennTreebank(split='valid'))
test_data = data_process(PennTreebank(split='test'))

# Batching


def batchify(data, bsz):
    # Concatenate all sequences
    data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    # Compute number of batches
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


batch_size = 32
eval_batch_size = 32

train_data = batchify(train_data, batch_size)
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

# Model parameters
input_size = len(vocab)
hidden_size = 100
output_size = len(vocab)
num_epochs = 3
learning_rate = 0.005

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move data to device
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

# Training function


def train_model(model, train_data, val_data, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.
        total_count = 0
        for batch, i in enumerate(range(0, train_data.size(0) - 1, 35)):
            data, targets = get_batch(train_data, i)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output.view(-1, output_size), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item() * targets.numel()
            total_count += targets.numel()

            if batch % 20 == 0 and batch > 0:
                cur_loss = total_loss / total_count
                print(f'| epoch {epoch:3d} | {batch:5d}/{train_data.size(0) // 35:5d} batches | '
                      f'loss {cur_loss:5.2f}')
                total_loss = 0
                total_count = 0

        val_loss = evaluate(model, val_data)
        print(f'| end of epoch {epoch:3d} | valid loss {val_loss:5.2f}')

# Evaluation function


def evaluate(model, data_source):
    model.eval()
    total_loss = 0.
    total_count = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, 35):
            data, targets = get_batch(data_source, i)
            output, _ = model(data)
            output_flat = output.view(-1, output_size)
            targets_flat = targets.view(-1)
            loss = criterion(output_flat, targets_flat)
            total_loss += loss.item() * targets_flat.size(0)
            total_count += targets_flat.size(0)
    return total_loss / total_count


def get_batch(source, i):
    seq_len = min(35, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)

    # Convert to one-hot encoding
    one_hot_data = torch.zeros(seq_len, batch_size, input_size, device=device)
    one_hot_data.scatter_(2, data.unsqueeze(2), 1)

    return one_hot_data, target


# Test both models
rnn_model = RNNNet(input_size, hidden_size, output_size).to(device)
transform_rnn_model = TransformRNNNet(
    input_size, hidden_size, output_size, train_transform_increment_coeff=True).to(device)

print("\nTraining TransformRNN model...")
train_model(transform_rnn_model, train_data,
            val_data, num_epochs, learning_rate)
transform_rnn_loss = evaluate(transform_rnn_model, test_data)
print(f"TransformRNN Test Loss: {transform_rnn_loss:.4f}")

print("Training RNN model...")
train_model(rnn_model, train_data, val_data, num_epochs, learning_rate)
rnn_loss = evaluate(rnn_model, test_data)
print(f"RNN Test Loss: {rnn_loss:.4f}")


# Generate some text using the trained models


def generate_text(model, seed_text, length=50):
    model.eval()
    with torch.no_grad():
        words = tokenizer(seed_text)
        input_ids = torch.tensor([vocab[w]
                                 for w in words], dtype=torch.long).to(device)
        generated_words = list(words)
        for _ in range(length):
            # Convert to one-hot encoding
            one_hot_input = torch.zeros(
                1, len(input_ids), input_size, device=device)
            one_hot_input.scatter_(2, input_ids.unsqueeze(0).unsqueeze(2), 1)

            output, _ = model(one_hot_input)
            word_weights = output[0, -1].div(0.8).exp()
            word_idx = torch.multinomial(word_weights, 1)[0]
            generated_words.append(vocab.lookup_token(word_idx.item()))
            input_ids = torch.cat(
                [input_ids[1:], word_idx.unsqueeze(0)], dim=0)
    return ' '.join(generated_words)


print("\nGenerating text with RNN model:")
print(generate_text(rnn_model, "the market"))

print("\nGenerating text with TransformRNN model:")
print(generate_text(transform_rnn_model, "the market"))
