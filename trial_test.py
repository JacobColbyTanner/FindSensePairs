import neurogym as ngym
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now import neurogym

# Register the ContextObjectStimuli task
register(
    id='ContextObjectStimuli-v0',
    entry_point='ContextObjectStimuli:ContextObjectStimuli'
)


def context_function_0(obj):
    return 1 if obj == 0 else 0


def context_function_1(obj):
    return 1 if obj == 1 else 0


# Create the dataset
num_contexts = 2
num_objects = 2
num_actions = 1
batch_size = 16
seq_len = 100

kwargs = {
    'num_contexts': num_contexts,
    'num_objects': num_objects,
    'num_actions': num_actions,
    'context_functions': [context_function_0, context_function_1],
    'context_transition_trial': False
}

dataset = ngym.Dataset('ContextObjectStimuli-v0',
                       env_kwargs=kwargs, batch_size=batch_size, seq_len=seq_len)


def plot_dataset(inputs, labels):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot inputs
    im = ax1.imshow(inputs.T, aspect='auto', cmap='viridis')
    ax1.set_ylabel('Input dimension')
    ax1.set_title('Inputs')
    plt.colorbar(im, ax=ax1)

    # Plot labels
    ax2.plot(labels, label='Labels', color='red')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Label')
    ax2.set_title('Labels')
    ax2.legend()

    plt.tight_layout()
    plt.show()


# Generate and plot a few batches
for _ in range(3):
    inputs, labels = dataset()

    # Select a single trial from the batch
    trial_idx = np.random.randint(batch_size)
    inputs_single = inputs[:, trial_idx, :]
    labels_single = labels[:, trial_idx]

    plot_dataset(inputs_single, labels_single)
    print(f"Trial length: {len(inputs_single)} steps")
    print(f"Unique labels: {np.unique(labels_single)}")
