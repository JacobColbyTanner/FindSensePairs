import torch
import torch.nn as nn
import torch.optim as optim
from CTRNN import RNNNet, TransformRNNNet, TransformRNNNet2, ContextTransformRNNNet, ContextRNNNet, ActionMapRNNNet, OutputMapRNNNet, HiddenMapRNNNet, InputMapRNNNet
import numpy as np
import requests
import os
import json
from torch.utils.data import TensorDataset, DataLoader, random_split

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Download TinyShakespeare dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text

# Character-level tokenization
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

# Convert text to indices
data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

# Prepare dataset


def create_sequence_pairs(data, seq_length):
    input_sequences = []
    target_sequences = []
    for i in range(0, len(data) - seq_length - 1, seq_length):
        input_seq = data[i:i+seq_length]
        target_seq = data[i+1:i+seq_length+1]
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)
    return torch.stack(input_sequences), torch.stack(target_sequences)


sequence_length = 8
batch_size = 32  # Increased batch size to match original

# Create input-target sequence pairs
input_sequences, target_sequences = create_sequence_pairs(
    data, sequence_length)

# Create TensorDataset
dataset = TensorDataset(input_sequences, target_sequences)

# Split into train, validation, and test sets
train_size = int(0.9 * len(dataset))

train_dataset, val_dataset = random_split(
    dataset, [train_size, len(dataset) - train_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model parameters
input_size = vocab_size
hidden_size = 10
transform_context_size = 10
transform_hidden_size = 20
output_size = vocab_size
num_epochs = 5
transform_learning_rate = 0.01
learning_rate = 0.001
num_steps = 1  # Number of steps per prediction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_batch(batch):
    input_seq, target_seq = batch
    # Convert to one-hot encoding
    one_hot_input = torch.zeros(input_seq.size(
        0), input_seq.size(1), input_size, device=device)
    one_hot_input.scatter_(2, input_seq.unsqueeze(2), 1)
    return one_hot_input.to(device), target_seq.to(device)


def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.
    total_count = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in data_loader:
            data, targets = get_batch(batch)
            output, _ = model(data, num_steps=num_steps)
            output_flat = output.view(-1, output_size)
            total_loss += criterion(output_flat,
                                    targets.view(-1)).item() * targets.numel()
            total_count += targets.numel()
    return total_loss / total_count


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.
        total_count = 0
        for batch, (data, targets) in enumerate(train_loader):
            data, targets = get_batch((data, targets))
            optimizer.zero_grad()
            output, _ = model(data, num_steps=num_steps)
            loss = criterion(output.view(-1, output_size), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item() * targets.numel()
            total_count += targets.numel()

            if batch % 100 == 0 and batch > 0:
                cur_loss = total_loss / total_count
                print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                      f'loss {cur_loss:5.2f}')
                total_loss = 0
                total_count = 0

        cur_loss = total_loss / total_count
        print(f'| epoch {epoch:3d} | train loss {cur_loss:5.2f}')

        val_loss = evaluate(model, val_loader)
        print(f'| end of epoch {epoch:3d} | valid loss {val_loss:5.2f}')

        # Save the model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            save_model(model, optimizer, epoch, val_loss, model_name)

    # Load the best model before returning
    model.load_state_dict(best_model)
    return best_val_loss, model


def generate_text(model, seed_text, length=1000, greedy=False, temperature=0.8):
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([char_to_idx[ch]
                                 for ch in seed_text], dtype=torch.long).to(device)
        generated_chars = list(seed_text)
        for _ in range(length):
            # Convert to one-hot encoding
            one_hot_input = torch.zeros(
                1, len(input_ids), input_size, device=device)
            one_hot_input.scatter_(2, input_ids.unsqueeze(0).unsqueeze(2), 1)

            output, _ = model(one_hot_input, num_steps=num_steps)

            if greedy:
                char_idx = output[0, -1].argmax().item()
            else:
                char_weights = output[0, -1].div(temperature).exp()
                char_idx = torch.multinomial(char_weights, 1)[0].item()

            generated_chars.append(idx_to_char[char_idx])
            input_ids = torch.cat(
                [input_ids[1:], torch.tensor([char_idx], device=device)], dim=0)
    return ''.join(generated_chars)


def save_model(model, optimizer, epoch, val_loss, model_name):
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(state, f'saved_models/{model_name}_model.pth')

    # Save hyperparameters and other necessary information
    params = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'num_steps': num_steps,
    }
    with open(f'saved_models/{model_name}_params.json', 'w') as f:
        json.dump(params, f)


def load_model(model_class, model_name):
    # Load hyperparameters and other necessary information
    with open(f'saved_models/{model_name}_params.json', 'r') as f:
        params = json.load(f)

    model = model_class(
        params['input_size'], params['hidden_size'], params['output_size']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    checkpoint = torch.load(f'saved_models/{model_name}_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']

    return model, optimizer, epoch, val_loss, params


# Test both models
rnn_model = RNNNet(input_size, hidden_size, output_size).to(device)
action_map_rnn_model = OutputMapRNNNet(
    input_size, transform_context_size, transform_hidden_size, output_size).to(device)

# print the number of parameters in each model
print(
    f"Number of parameters in RNN model: {sum(p.numel() for p in rnn_model.parameters())}")
print(
    f"Number of parameters in ActionMapRNN model: {sum(p.numel() for p in action_map_rnn_model.parameters())}")


print("\nTraining ActionMapRNN model...")
action_map_rnn_loss, action_map_rnn_model = train_model(
    action_map_rnn_model, train_loader, val_loader, num_epochs, transform_learning_rate, "action_map_rnn")
print(f"ActionMapRNN Best Validation Loss: {action_map_rnn_loss:.4f}")

print("\nTraining RNN model...")
rnn_loss, rnn_model = train_model(rnn_model, train_loader, val_loader,
                                  num_epochs, learning_rate, "rnn")
print(f"RNN Best Validation Loss: {rnn_loss:.4f}")


print("\n\nGenerating text with ActionMapRNN model at temperature 0.8:")
print(generate_text(action_map_rnn_model, "To be or not to be", temperature=0.8))

print("\n\nGenerating text with ActionMapRNN model at temperature 0.5:")
print(generate_text(action_map_rnn_model, "To be or not to be", temperature=0.5))

print("\n\nGenerating text with ActionMapRNN model at temperature 0.3:")
print(generate_text(action_map_rnn_model, "To be or not to be", temperature=0.3))

print("\n\nGenerating text with RNN model at temperature 0.8:")
print(generate_text(rnn_model, "To be or not to be", temperature=0.8))

print("\n\nGenerating text with RNN model at temperature 0.5:")
print(generate_text(rnn_model, "To be or not to be", temperature=0.5))

print("\n\nGenerating text with RNN model at temperature 0.3:")
print(generate_text(rnn_model, "To be or not to be", temperature=0.3))

# Example of how to load and continue training or evaluate a saved model
# loaded_rnn_model, loaded_optimizer, start_epoch, best_val_loss, params = load_model(RNNNet, "rnn")
# continue training or evaluate as needed
