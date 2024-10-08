from CTRNN import RNNNet, TransformRNNNet, ActionMapRNNNet, OutputMapRNNNet, HiddenMapRNNNet
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.envs.registration import register
import neurogym as ngym
import matplotlib.pyplot as plt
import pickle
import datetime

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Register the ContextObjectStimuli task
register(
    id='ContextObjectStimuli-v0',
    entry_point='ContextObjectStimuli:ContextObjectStimuli'
)

# Set this flag to True to use the TransformRNNNet, False to use the regular RNNNet
USE_TRANSFORM_RNN = False


def create_object_selector_functions(num_contexts, num_objects):
    return [
        {obj: 1 if obj == context else 0 for obj in range(num_objects)}
        for context in range(num_contexts)
    ]


def train_rnn(net, dataset, criterion, optimizer, num_epochs, batch_size):
    running_loss = 0
    for epoch in range(num_epochs):
        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).float()
        labels = torch.from_numpy(labels).long()

        optimizer.zero_grad()
        outputs, _ = net(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            avg_loss = running_loss / 100
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
            running_loss = 0

            # Calculate accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs, 2)
                correct = (predicted == labels).sum().item()
                total = labels.numel()
                accuracy = correct / total * 100

                # Calculate baseline accuracy (most common action)
                unique, counts = torch.unique(labels, return_counts=True)
                baseline_accuracy = counts.max().item() / total * 100

                # Calculate scaled accuracy
                scaled_accuracy = (accuracy - baseline_accuracy) / \
                    (100 - baseline_accuracy) * 100

                print(f'Raw Accuracy: {accuracy:.2f}%')
                print(f'Baseline Accuracy: {baseline_accuracy:.2f}%')
                print(f'Scaled Accuracy: {scaled_accuracy:.2f}%')

    return net


def plot_network_performance(net, dataset):
    inputs, labels = dataset()
    inputs = torch.from_numpy(inputs).float()

    with torch.no_grad():
        outputs, _ = net(inputs)
        _, predicted = torch.max(outputs, 2)

    # Select a single trial from the batch
    trial_idx = np.random.randint(inputs.shape[1])
    inputs_single = inputs[:, trial_idx, :].numpy()
    labels_single = labels[:, trial_idx]
    predicted_single = predicted[:, trial_idx].numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot observations
    im = ax1.imshow(inputs_single.T, aspect='auto', cmap='viridis')
    ax1.set_ylabel('Observation dimensions')
    ax1.set_title('Single Trial Observations')
    plt.colorbar(im, ax=ax1)

    # Plot ground truth and predicted actions
    time_steps = range(len(labels_single))
    ax2.step(time_steps, labels_single, where='post',
             label='Ground Truth', linestyle='--', color='blue')
    ax2.step(time_steps, predicted_single, where='post',
             label='Predicted', linestyle='-', color='red')
    ax2.set_xlabel('Time steps')
    ax2.set_ylabel('Action')
    ax2.set_title('Ground Truth vs Predicted Actions')
    ax2.legend()

    # Set y-axis limits to show all possible actions
    ax2.set_ylim(-0.5, max(np.max(labels_single),
                 np.max(predicted_single)) + 0.5)

    plt.tight_layout()
    plt.show()


def save_agent(net, env_params, model_params):
    # Create a timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the saved_agents directory if it doesn't exist
    save_dir = 'saved_agents'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Prepare the data to be saved
    save_data = {
        'model_state_dict': net.state_dict(),
        'env_params': env_params,
        'model_params': model_params
    }

    # Save the data
    filename = f"{save_dir}/agent_{timestamp}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"Agent saved to {filename}")


def main():
    # Hyperparameters
    num_contexts = 2
    num_objects = 2
    num_actions = 1
    batch_size = 64
    hidden_size = 3
    learning_rate = 1e-2
    num_epochs = 1000

    # Create the object selector functions
    context_functions = create_object_selector_functions(
        num_contexts, num_objects)

    # Create the dataset
    env_params = {
        'num_contexts': num_contexts,
        'num_objects': num_objects,
        'num_actions': num_actions,
        'context_functions': context_functions,
        'trial_type': "object_transition"
    }

    # Create the environment first to access its timing information
    env = ngym.make('ContextObjectStimuli-v0', **env_params)

    # Calculate seq_len based on the environment's timing
    seq_len = 3*sum(env.timing.values()) // env.dt
    print(f"Sequence length: {seq_len} time steps")

    dataset = ngym.Dataset(env, batch_size=batch_size, seq_len=seq_len)

    # Get input and output sizes
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    # Create the RNN
    if USE_TRANSFORM_RNN:
        net = OutputMapRNNNet(input_size, hidden_size, 10,
                              output_size, use_tanh=True)
        print("Using OutputMapRNNNet")
    else:
        net = RNNNet(input_size, hidden_size, output_size)
        print("Using RNNNet")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Train the RNN
    net = train_rnn(net, dataset, criterion, optimizer, num_epochs, batch_size)

    # Save the trained agent
    model_params = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'use_transform_rnn': USE_TRANSFORM_RNN
    }
    save_agent(net, env_params, model_params)

    # Plot network performance after training
    plot_network_performance(net, dataset)


if __name__ == "__main__":
    main()
