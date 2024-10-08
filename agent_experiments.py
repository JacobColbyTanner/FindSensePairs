import os
import pickle
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from fastdtw import fastdtw
from sklearn.cluster import SpectralClustering
# Update this import
from CTRNN import RNNNet, TransformRNNNet, ActionMapRNNNet, OutputMapRNNNet, HiddenMapRNNNet
import neurogym as ngym
from gym.envs.registration import register
from scipy import linalg
import networkx as nx
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

# Import the ContextObjectStimuli environment
from ContextObjectStimuli import ContextObjectStimuli

# Register the ContextObjectStimuli environment
register(
    id='ContextObjectStimuli-v0',
    entry_point='ContextObjectStimuli:ContextObjectStimuli'
)


def load_agent(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    if data['model_params'].get('use_transform_rnn', False):
        print("Using TransformRNNNet")
        print(data['model_params'])
        net = OutputMapRNNNet(data['model_params']['input_size'],
                              data['model_params']['hidden_size'],
                              10,
                              data['model_params']['output_size'])
    else:
        net = RNNNet(data['model_params']['input_size'],
                     data['model_params']['hidden_size'],
                     data['model_params']['output_size'])
    net.load_state_dict(data['model_state_dict'])

    return net, data['env_params'], data['model_params']


def generate_trial_dataset(env, num_trials_per_context, trial_length, object_duration):
    trials = []
    for context in range(env.num_contexts):
        # Ensure equal representation of objects in the first position
        first_objects = list(range(env.num_objects)) * \
            (num_trials_per_context // env.num_objects)
        first_objects += random.sample(range(env.num_objects),
                                       num_trials_per_context % env.num_objects)
        random.shuffle(first_objects)

        for first_object in first_objects:
            # Generate a sequence of objects without repetition
            remaining_objects = list(range(env.num_objects))
            remaining_objects.remove(first_object)
            object_sequence = [first_object]

            for _ in range(trial_length - 1):
                next_object = random.choice(
                    [obj for obj in remaining_objects if obj != object_sequence[-1]])
                object_sequence.append(next_object)
                remaining_objects = [
                    obj for obj in remaining_objects if obj != next_object]
                if not remaining_objects:
                    remaining_objects = [obj for obj in range(
                        env.num_objects) if obj != object_sequence[-1]]

            trial = []
            for obj in object_sequence:
                obs = np.zeros(env.num_contexts + env.num_objects)
                obs[context] = 1
                obs[env.num_contexts + obj] = 1
                trial.extend([obs] * object_duration)
            trials.append((context, first_object, np.array(trial)))

    return trials


def collect_hidden_states(net, trials):
    hidden_states = []
    with torch.no_grad():
        for _, _, trial in trials:
            inputs = torch.from_numpy(trial).float().unsqueeze(1)
            _, hidden = net(inputs)
            hidden_states.append(hidden.squeeze().numpy())
    return hidden_states


def compute_dtw_correlations(hidden_states):
    n = len(hidden_states)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance, _ = fastdtw(hidden_states[i], hidden_states[j])
            dist_matrix[i, j] = dist_matrix[j, i] = distance

    # Convert distances to correlations
    max_dist = np.max(dist_matrix)
    corr_matrix = 1 - dist_matrix / max_dist
    return corr_matrix


def compute_regular_correlations(hidden_states):
    # Flatten each hidden state sequence
    flattened_states = [hs.flatten() for hs in hidden_states]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(flattened_states)

    return corr_matrix


def plot_correlation_matrix(corr_matrix, trials, title, filename):
    # Compute linkage matrix
    linkage = hierarchy.ward(pdist(corr_matrix))

    # Get the order of trials for sorting
    order = hierarchy.dendrogram(linkage, no_plot=True)['leaves']

    # Sort the correlation matrix and trials
    sorted_corr_matrix = corr_matrix[order][:, order]
    sorted_trials = [trials[i] for i in order]

    plt.figure(figsize=(12, 10))
    plt.imshow(sorted_corr_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()

    # Label axes
    contexts = [f"C{t[0]}" for t in sorted_trials]
    objects = [f"O{t[1]}" for t in sorted_trials]
    labels = [f"{c}-{o}" for c, o in zip(contexts, objects)]

    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    return order  # Return the sorting order for potential further use


def compute_modularity(corr_matrix, labels):
    k = np.sum(corr_matrix, axis=1)
    m = np.sum(k)
    Q = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] == labels[j]:
                Q += corr_matrix[i, j] - k[i] * k[j] / m
    return Q / m


def effective_num_communities(corr_matrix):
    # Compute the graph Laplacian
    D = np.diag(np.sum(corr_matrix, axis=1))
    L = D - corr_matrix

    # Compute eigenvalues of the Laplacian
    eigenvalues = linalg.eigvalsh(L)

    # Sort eigenvalues in ascending order
    eigenvalues.sort()

    # Find the largest gap in the eigenvalue spectrum
    gaps = np.diff(eigenvalues)
    largest_gap = np.argmax(gaps) + 1

    return largest_gap


def compute_accuracy(net, trials, env):
    correct = 0
    total = 0
    with torch.no_grad():
        for context, first_object, trial_data in trials:
            inputs = torch.from_numpy(trial_data).float().unsqueeze(1)
            outputs, _ = net(inputs)
            _, predicted = torch.max(outputs, 2)

            # Compute ground truth
            ground_truth = []
            for step in trial_data:
                object_index = np.argmax(step[env.num_contexts:])
                correct_action = env.context_functions[context][object_index]
                ground_truth.append(correct_action)

            ground_truth = torch.tensor(ground_truth).long()

            correct += (predicted.squeeze() == ground_truth).sum().item()
            total += len(ground_truth)

    accuracy = correct / total * 100
    return accuracy


def compute_connected_components(corr_matrix, thresholds):
    components = []
    for threshold in thresholds:
        # Create a graph from the thresholded correlation matrix
        G = nx.Graph()
        n = corr_matrix.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                if corr_matrix[i, j] >= threshold:
                    G.add_edge(i, j)

        # Compute the number of connected components
        num_components = nx.number_connected_components(G)
        components.append(num_components)

    return components


def eigengap_communities(corr_matrix):
    # Compute the graph Laplacian
    D = np.diag(np.sum(corr_matrix, axis=1))
    L = D - corr_matrix

    # Compute eigenvalues of the Laplacian
    eigenvalues = linalg.eigvalsh(L)

    # Sort eigenvalues in ascending order
    eigenvalues.sort()

    # Compute gaps
    gaps = np.diff(eigenvalues)

    # Return the index of the largest gap in the first n/2 eigenvalues
    n = len(eigenvalues)
    return np.argmax(gaps[:n//2]) + 1


def modularity_communities(corr_matrix, max_communities=10):
    best_modularity = -np.inf
    best_n_communities = 0

    for n in range(2, max_communities + 1):
        clustering = SpectralClustering(
            n_clusters=n, affinity='precomputed', random_state=0)
        labels = clustering.fit_predict(corr_matrix)

        modularity = compute_modularity(corr_matrix, labels)

        if modularity > best_modularity:
            best_modularity = modularity
            best_n_communities = n

    return best_n_communities


def plot_modularity_curve(corr_matrix, max_communities=10, rnn_type="RNN"):
    modularities = []
    for n in range(2, max_communities + 1):
        clustering = SpectralClustering(
            n_clusters=n, affinity='precomputed', random_state=0)
        labels = clustering.fit_predict(corr_matrix)
        modularity = compute_modularity(corr_matrix, labels)
        modularities.append(modularity)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_communities + 1), modularities, marker='o')
    plt.xlabel('Number of Communities')
    plt.ylabel('Modularity')
    plt.title(f'Modularity vs Number of Communities ({rnn_type})')
    plt.savefig(f'modularity_curve_{rnn_type}.png')
    plt.close()


def main():
    # Load the latest agent
    agent_files = sorted(os.listdir('saved_agents'), reverse=True)
    if not agent_files:
        print("No saved agents found.")
        return

    latest_agent = os.path.join('saved_agents', agent_files[0])
    net, env_params, model_params = load_agent(latest_agent)

    # Determine the RNN type
    rnn_type = "TransformRNN" if model_params.get(
        'use_transform_rnn', False) else "RNN"

    # Create environment
    env = ngym.make('ContextObjectStimuli-v0', **env_params)

    print(f"Number of contexts: {env.num_contexts}")
    print(f"Number of objects: {env.num_objects}")
    print(f"Context functions: {env.context_functions}")

    # Generate trial dataset
    num_trials_per_context = 1*env_params['num_contexts']
    trial_length = 1*env_params['num_objects']
    object_duration = 10
    trials = generate_trial_dataset(
        env, num_trials_per_context, trial_length, object_duration)

    # Collect hidden states
    hidden_states = collect_hidden_states(net, trials)

    # Remove the first trial for each context
    num_contexts = env.num_contexts
    trials_to_keep = [i for i in range(
        len(trials)) if i % num_trials_per_context != 0]
    trials = [trials[i] for i in trials_to_keep]
    hidden_states = [hidden_states[i] for i in trials_to_keep]

    # Compute accuracy
    accuracy = compute_accuracy(net, trials, env)
    print(f"Accuracy on generated trials: {accuracy:.2f}%")

    # Compute baseline accuracy (most common action)
    all_actions = [env.context_functions[context][obj] for context in range(
        env.num_contexts) for obj in range(env.num_objects)]
    most_common_action = max(set(all_actions), key=all_actions.count)
    baseline_accuracy = all_actions.count(
        most_common_action) / len(all_actions) * 100
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")

    # Compute scaled accuracy
    scaled_accuracy = (accuracy - baseline_accuracy) / \
        (100 - baseline_accuracy) * 100
    print(f"Scaled Accuracy: {scaled_accuracy:.2f}%")

    # Compute DTW correlations
    dtw_corr_matrix = compute_dtw_correlations(hidden_states)

    # Compute regular correlations
    reg_corr_matrix = compute_regular_correlations(hidden_states)

    # Plot correlation matrices
    dtw_order = plot_correlation_matrix(
        dtw_corr_matrix, trials, f"DTW Correlation Matrix ({rnn_type}, Sorted)", f"dtw_correlation_matrix_{rnn_type}_sorted.png")
    reg_order = plot_correlation_matrix(
        reg_corr_matrix, trials, f"Regular Correlation Matrix ({rnn_type}, Sorted)", f"regular_correlation_matrix_{rnn_type}_sorted.png")

    # Compute effective number of communities using different methods
    dtw_eigengap = eigengap_communities(dtw_corr_matrix)
    reg_eigengap = eigengap_communities(reg_corr_matrix)

    dtw_modularity = modularity_communities(dtw_corr_matrix)
    reg_modularity = modularity_communities(reg_corr_matrix)

    print(f"DTW Eigengap Communities: {dtw_eigengap}")
    print(f"Regular Eigengap Communities: {reg_eigengap}")
    print(f"DTW Modularity Communities: {dtw_modularity}")
    print(f"Regular Modularity Communities: {reg_modularity}")

    # Compute connected components for different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    dtw_components = compute_connected_components(dtw_corr_matrix, thresholds)
    reg_components = compute_connected_components(reg_corr_matrix, thresholds)

    print("\nNumber of connected components:")
    print("Threshold | DTW | Regular")
    print("----------------------------")
    for t, dtw_c, reg_c in zip(thresholds, dtw_components, reg_components):
        print(f"{t:.1f}      | {dtw_c:3d} | {reg_c:3d}")

    # Plot modularity curves
    plot_modularity_curve(dtw_corr_matrix, rnn_type=rnn_type)


if __name__ == "__main__":
    main()
