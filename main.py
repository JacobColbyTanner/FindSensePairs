
# Define networks

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from gym.envs.registration import register
import torch.optim as optim
import gym
from CTRNN import RNNNet, CTRNN
import math
from torch.nn import functional as F
from torch.nn import init
import torch.nn as nn
import torch
import neurogym as ngym
import sys
import os

# Add the path to your local neurogym package
local_neurogym_path = os.path.abspath(
    '/Users/jacobtanner/neurogym_match_to_discrete/neurogym')
if local_neurogym_path not in sys.path:
    sys.path.insert(0, local_neurogym_path)

# Now import neurogym


register(id='FindSensePairs-v0', entry_point="FindSensePairs:FindSensePairs")


# Replace 'YourEnvName-v0' with the ID you used during registration
env = gym.make('FindSensePairs-v0')


def get_accuracy(net):
    env.reset(no_step=True)
    perf = 0
    num_trial = 100
    correct_predictions = 0
    total_predictions = 0
    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)
        action_pred, rnn_activity = net(inputs)

        # Calculate accuracy
        predicted_actions = torch.argmax(
            action_pred[:, 0, :], dim=-1).detach().numpy()
        correct_predictions += np.sum(predicted_actions == gt)
        total_predictions += gt.size

    # Calculate overall accuracy
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# hyperparameters
N = 60
max_iters = 10000
batch_size = 128
learning_rate = 1e-3
# Environment
task = 'FindSensePairs-v0'  # 'PerceptualDecisionMaking-v0'
kwargs = {'dt': 100, 'N': N, 'interpolate_delay': 0}
# kwargs = {'dt': 100}
seq_len = 100


# Make supervised dataset
dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=batch_size,
                       seq_len=seq_len)

# A sample environment from dataset
env = dataset.env


# Network input and output size
input_size = env.observation_space.shape[0]
output_size = env.action_space.n


# Instantiate the network and print information
hidden_size = 64
net = RNNNet(input_size=input_size, hidden_size=hidden_size,
             output_size=output_size, dt=env.dt)
print(net)

# Use Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

running_loss = 0
running_acc = 0
correct_predictions = 0
total_predictions = 0

inputs, labels_orig = dataset()

plt.figure()
plt.imshow(inputs[:, 0, :].squeeze().T)
plt.show()

for i in range(max_iters):
    inputs, labels_orig = dataset()
    inputs = torch.from_numpy(inputs).type(torch.float)
    labels = torch.from_numpy(labels_orig.flatten()).type(torch.long)

    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output, _ = net(inputs)

    output = output.view(-1, output_size)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()    # Does the update

    running_loss += loss.item()
    if i % 100 == 99:
        running_loss /= 100
        print('Step {}, Loss {:0.4f}'.format(i+1, running_loss))
        running_loss = 0
        # Calculate overall accuracy
        accuracy = get_accuracy(net)


net.eval()
env.reset(no_step=True)
perf = 0
num_trial = 100
activity_dict = {}
trial_infos = {}
all_inputs = []

pairs = []
N_div2 = int(N/2)
for i in range(N_div2):
    for j in range(N_div2):
        pairs.append((i+1, j+N_div2+1))


# Concatenate activity for PCA
activity = np.concatenate(
    list(activity_dict[i] for i in range(num_trial)), axis=0)
print('Shape of the neural activity: (Time points, Neurons): ', activity.shape)


plt.figure()
plt.imshow(inputs.detach().numpy().squeeze().T)
plt.colorbar()
plt.show()

# Compute PCA and visualize


pca = PCA(n_components=3)
pca.fit(activity)

plt.figure()
for i in range(num_trial):
    activity_pc = pca.transform(activity_dict[i])
    trial = trial_infos[i]
    color = 'red' if trial['ground_truth'] == 0 else 'blue'
    plt.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color)

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(num_trial):
    activity_pc = pca.transform(activity_dict[i])
    trial = trial_infos[i]
    if trial_infos[i]['pair'][0] == 1:
        color = 'red'
    elif trial_infos[i]['pair'][0] == 2:
        color = 'green'
    elif trial_infos[i]['pair'][0] == 3:
        color = 'black'
    elif trial_infos[i]['pair'][0] == 4:
        color = 'yellow'
    elif trial['ground_truth'] == 0:
        color = 'blue'

    ax.plot(activity_pc[:, 0], activity_pc[:, 1],
            activity_pc[:, 2], 'o-', color=color)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.show()


pairs = []
for i in range(num_trial):
    pairs.append(trial_infos[i]['pair'])

pairs = np.array(pairs)

print(pairs.shape)

# index where first column of pairs is 1 and second column is 3
idx = pairs[:, 0] == 1
idx2 = pairs[:, 1] == 3
# find the first instance where both are true
idx = idx & idx2
print(idx)
idx = np.argmax(idx)

# find first nonzero in idd (a Boolean array)

idxx = np.nonzero(idx)

# activity_pc = pca.transform(activity_dict[idxx])

# Make subplots with plt.subplots
fig, axs = plt.subplots(4)
axs[0].imshow(all_inputs[:, idxx[0]].T)
axs[1].imshow(activity_dict[:, idxx[0]].T)


# index where first column of pairs is 1 and second column is 3
idx = pairs[:, 0] == 2
idx2 = pairs[:, 1] == 4
# find the first instance where both are true
idx = idx & idx2
print(idx)
idx = np.argmax(idx)


# find first nonzero in idd (a Boolean array)
print(idx)
idyy = np.nonzero(idx)

# activity_pc = pca.transform(activity_dict[idxx])


axs[2].imshow(all_inputs[:, idyy[0]].T)
axs[3].imshow(activity_dict[:, idyy[0]].T)
plt.show()
