from sklearn.decomposition import PCA
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
from CTRNN import RNNNet, TransformRNNNet, ActionMapRNNNet, OutputMapRNNNet, HiddenMapRNNNet, ContextInputSubspaceRNNNet, ActionEmbeddingActionMapNet
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


def replace_inf_nan(matrix, inf_replacement=1e6, nan_replacement=-1):
    matrix = np.where(np.isinf(matrix), inf_replacement, matrix)
    matrix = np.where(np.isnan(matrix), nan_replacement, matrix)
    return matrix


def load_agent(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    if data['model_params'].get('use_transform_rnn', False):
        print("Using TransformRNNNet")
        print(data['model_params'])
        net = ActionEmbeddingActionMapNet(data['model_params']['input_size'],
                                          data['model_params']['hidden_size'],
                                          data['model_params']['output_size'])
    else:
        net = RNNNet(data['model_params']['input_size'],
                     data['model_params']['hidden_size'],
                     data['model_params']['output_size'])
    net.load_state_dict(data['model_state_dict'])

    return net, data['env_params'], data['model_params']


def generate_action_cycles(valid_actions):
    num_actions = len(valid_actions)
    cycles = [valid_actions]

    for i in range(1, num_actions):
        new_cycle = valid_actions[i:] + valid_actions[:i]
        cycles.append(new_cycle)

    return cycles


def generate_trial_dataset(env, num_trials_per_action=1, num_cycle_repetitions=1, object_presentation_time=10, context_first_object_only=False):
    trials = []

    for context in range(env.num_contexts):
        context_function = env.context_functions[context]

        # Find valid actions for this context
        valid_actions = sorted(set(context_function.values()))

        # Pad the valid actions
        padded_actions = []
        for i in range(env.num_actions):
            if i in valid_actions:
                padded_actions.append(i)
            else:
                if len(padded_actions) > 0:
                    padded_actions.append(padded_actions[-1])
                else:
                    padded_actions.append(min(valid_actions))

        # Generate action cycles for this context
        action_cycles = generate_action_cycles(padded_actions)

        for action_cycle in action_cycles:
            for _ in range(num_trials_per_action):
                # Repeat the action cycle
                full_action_sequence = action_cycle * num_cycle_repetitions

                # Generate the trial data
                trial = []
                for i, action in enumerate(full_action_sequence):
                    # Find objects that map to this action in the current context
                    valid_objects = [
                        obj for obj, act in context_function.items() if act == action]

                    # Choose a random object from valid objects
                    chosen_object = random.choice(valid_objects)

                    # Create the observation
                    obs = np.zeros(env.num_contexts + env.num_objects)
                    if not context_first_object_only or i == 0:
                        obs[context] = 1
                    obs[env.num_contexts + chosen_object] = 1

                    # Repeat the observation for object_presentation_time steps
                    trial.extend([obs] * object_presentation_time)

                trials.append((context, action_cycle[0], np.array(trial)))

    return trials


def collect_hidden_states(net, trials):
    hidden_state_trajectories = []
    with torch.no_grad():
        for _, _, trial in trials:
            inputs = torch.from_numpy(trial).float().unsqueeze(1)
            hidden = None
            trial_trajectory = []
            for step in range(inputs.size(0)):
                _, hidden = net.rnn(inputs[step:step+1], hidden)
                trial_trajectory.append(hidden.squeeze().numpy())
            hidden_state_trajectories.append(np.array(trial_trajectory))
    return hidden_state_trajectories


def compute_dtw_correlations(hidden_state_trajectories):
    n = len(hidden_state_trajectories)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            # Compute DTW distance for each dimension and average
            distances = []
            for dim in range(hidden_state_trajectories[i].shape[1]):
                distance, _ = fastdtw(hidden_state_trajectories[i][:, dim],
                                      hidden_state_trajectories[j][:, dim])
                distances.append(distance)
            avg_distance = np.mean(distances)
            dist_matrix[i, j] = dist_matrix[j, i] = avg_distance

    # Replace inf and nan values
    dist_matrix = replace_inf_nan(dist_matrix)

    # Convert distances to correlations
    max_dist = np.max(dist_matrix)
    corr_matrix = 1 - dist_matrix / max_dist
    return corr_matrix


def compute_regular_correlations(hidden_state_trajectories):
    n = len(hidden_state_trajectories)
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            # Compute correlation for each dimension
            correlations = []
            for dim in range(hidden_state_trajectories[i].shape[1]):
                traj_i = hidden_state_trajectories[i][:, dim]
                traj_j = hidden_state_trajectories[j][:, dim]
                corr = np.corrcoef(traj_i, traj_j)[0, 1]
                correlations.append(corr)

            # Average correlation across dimensions
            avg_corr = np.mean(correlations)
            corr_matrix[i, j] = corr_matrix[j, i] = avg_corr

    # Replace inf and nan values
    corr_matrix = replace_inf_nan(corr_matrix)

    return corr_matrix


def plot_correlation_matrix(corr_matrix, trials, title, filename):
    # Replace inf and nan values before computing linkage
    corr_matrix = replace_inf_nan(corr_matrix)

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
    # Replace inf and nan values
    corr_matrix = replace_inf_nan(corr_matrix)

    k = np.sum(corr_matrix, axis=1)
    m = np.sum(k)
    Q = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if labels[i] == labels[j]:
                Q += corr_matrix[i, j] - k[i] * k[j] / m
    return Q / m


def effective_num_communities(corr_matrix):
    # Replace inf and nan values
    corr_matrix = replace_inf_nan(corr_matrix)

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
    # Replace inf and nan values
    corr_matrix = replace_inf_nan(corr_matrix)

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
    # Replace inf and nan values
    corr_matrix = replace_inf_nan(corr_matrix)

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
    # Replace inf and nan values
    corr_matrix = replace_inf_nan(corr_matrix)

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
    # Replace inf and nan values
    corr_matrix = replace_inf_nan(corr_matrix)

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


def compute_euclidean_distances(hidden_state_trajectories):
    n = len(hidden_state_trajectories)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            # Compute Euclidean distance for each time step and average
            distances = np.linalg.norm(
                hidden_state_trajectories[i] - hidden_state_trajectories[j], axis=1)
            avg_distance = np.mean(distances)
            dist_matrix[i, j] = dist_matrix[j, i] = avg_distance

    # Replace inf and nan values
    dist_matrix = replace_inf_nan(dist_matrix)

    return dist_matrix


def plot_distance_matrix(dist_matrix, trials, title, filename):
    # Replace inf and nan values before computing linkage
    dist_matrix = replace_inf_nan(dist_matrix)

    # Compute linkage matrix
    linkage = hierarchy.ward(pdist(dist_matrix))

    # Get the order of trials for sorting
    order = hierarchy.dendrogram(linkage, no_plot=True)['leaves']

    # Sort the distance matrix and trials
    sorted_dist_matrix = dist_matrix[order][:, order]
    sorted_trials = [trials[i] for i in order]

    plt.figure(figsize=(12, 10))
    plt.imshow(sorted_dist_matrix, cmap='viridis_r', aspect='auto')
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

    return order


def plot_hidden_state_trajectories(net, env, trials, filename):
    plt.figure(figsize=(12, 10))

    # Create color maps for actions and contexts
    action_colors = plt.cm.rainbow(np.linspace(0, 1, env.num_actions))
    context_colors = plt.cm.viridis(np.linspace(0, 1, env.num_contexts))
    action_color_map = {action: color for action,
                        color in enumerate(action_colors)}
    context_color_map = {context: color for context,
                         color in enumerate(context_colors)}

    all_hidden_states = []

    with torch.no_grad():
        for trial in trials:
            context, first_action, trial_data = trial
            inputs = torch.from_numpy(trial_data).float().unsqueeze(1)

            # Initialize hidden state
            hidden = None
            hidden_states = []
            actions = []

            # Process the trial step by step
            for step in range(inputs.size(0)):
                _, hidden = net.rnn(inputs[step:step+1], hidden)
                hidden_states.append(hidden.squeeze().numpy())

                # Determine the current action based on the input
                object_index = np.argmax(trial_data[step, env.num_contexts:])
                current_action = env.context_functions[context][object_index]
                actions.append(current_action)

            hidden_states = np.array(hidden_states)
            all_hidden_states.extend(hidden_states)

    # Convert all hidden states to a numpy array
    all_hidden_states = np.array(all_hidden_states)

    # If hidden state dimension is greater than 2, use PCA
    if all_hidden_states.shape[1] > 2:
        pca = PCA(n_components=2)
        all_hidden_states_2d = pca.fit_transform(all_hidden_states)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    else:
        all_hidden_states_2d = all_hidden_states

    # Clip the magnitude of vectors if it exceeds the threshold
    magnitudes = np.linalg.norm(all_hidden_states_2d, axis=1)
    threshold = 10000  # np.min(magnitudes)
    scale_factors = np.minimum(threshold / magnitudes, 1)
    all_hidden_states_2d = all_hidden_states_2d * scale_factors[:, np.newaxis]

    # Plot trajectories
    start_idx = 0
    trial_ind = 0
    for trial in trials:
        trial_ind += 1
        # if trial_ind != 2 and trial_ind != 5:
        #    continue
        print(f"Plotting trial {trial_ind} out of {len(trials)}")

        context, first_action, trial_data = trial
        end_idx = start_idx + len(trial_data)

        hidden_states_2d = all_hidden_states_2d[start_idx:end_idx]
        actions = [env.context_functions[context][np.argmax(
            step[env.num_contexts:])] for step in trial_data]

        # Plot the trajectory lines (colored by context)
        plt.plot(hidden_states_2d[:, 0], hidden_states_2d[:, 1],
                 color=context_color_map[context], alpha=0.5, linewidth=1)

        # Plot points (colored by action)
        for i, (state, action) in enumerate(zip(hidden_states_2d, actions)):
            plt.scatter(state[0], state[1],
                        color=action_color_map[action], s=20, zorder=2)

        start_idx = end_idx

    plt.title(
        "Hidden State Trajectories (PCA)" if all_hidden_states.shape[1] > 2 else "Hidden State Trajectories")
    plt.xlabel(
        "PC1" if all_hidden_states.shape[1] > 2 else "Hidden Dimension 1")
    plt.ylabel(
        "PC2" if all_hidden_states.shape[1] > 2 else "Hidden Dimension 2")

    # Create legends for both actions and contexts
    action_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                markersize=10, label=f'Action {action}')
                     for action, color in action_color_map.items()]
    context_legend = [plt.Line2D([0], [0], color=color, label=f'Context {context}')
                      for context, color in context_color_map.items()]

    plt.legend(handles=action_legend + context_legend,
               loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
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
    # This will result in num_actions * 5 trials per context
    context_first_object_only = True
    num_trials_per_action = 5
    num_cycle_repetitions = 10
    object_presentation_time = 10
    trials = generate_trial_dataset(
        env, num_trials_per_action, num_cycle_repetitions, object_presentation_time, context_first_object_only=context_first_object_only)

    # Collect hidden states
    hidden_state_trajectories = collect_hidden_states(net, trials)

    # Remove the first trial for each context
    num_contexts = env.num_contexts

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
    dtw_corr_matrix = compute_dtw_correlations(hidden_state_trajectories)

    # Compute regular correlations
    reg_corr_matrix = compute_regular_correlations(hidden_state_trajectories)

    # Compute Euclidean distances
    euclidean_dist_matrix = compute_euclidean_distances(
        hidden_state_trajectories)

    # Plot correlation matrices
    dtw_order = plot_correlation_matrix(
        dtw_corr_matrix, trials, f"DTW Correlation Matrix ({rnn_type}, Sorted)", f"dtw_correlation_matrix_{rnn_type}_sorted.png")
    reg_order = plot_correlation_matrix(
        reg_corr_matrix, trials, f"Regular Correlation Matrix ({rnn_type}, Sorted)", f"regular_correlation_matrix_{rnn_type}_sorted.png")

    # Plot Euclidean distance matrix
    euclidean_order = plot_distance_matrix(
        euclidean_dist_matrix, trials, f"Euclidean Distance Matrix ({rnn_type}, Sorted)", f"euclidean_distance_matrix_{rnn_type}_sorted.png")

    # Compute effective number of communities using different methods
    # dtw_eigengap = eigengap_communities(dtw_corr_matrix)
    # reg_eigengap = eigengap_communities(reg_corr_matrix)

    dtw_modularity = modularity_communities(dtw_corr_matrix)
    # reg_modularity = modularity_communities(reg_corr_matrix)

    # print(f"DTW Eigengap Communities: {dtw_eigengap}")
    # print(f"Regular Eigengap Communities: {reg_eigengap}")
    print(f"DTW Modularity Communities: {dtw_modularity}")
    # print(f"Regular Modularity Communities: {reg_modularity}")

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

    # Generate trial dataset for trajectory plotting
    trajectory_trials = generate_trial_dataset(
        env, num_trials_per_action=1, num_cycle_repetitions=4, object_presentation_time=10, context_first_object_only=True)

    # Plot hidden state trajectories
    plot_hidden_state_trajectories(
        net, env, trajectory_trials, f"hidden_state_trajectories_{rnn_type}.png")


if __name__ == "__main__":
    main()
