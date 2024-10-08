# Define networks
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math


class CTRNN(nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms. 
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()

    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Run network for one time step.

        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)

        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        h_new = torch.relu(self.input2h(input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new

    def forward(self, input, hidden=None, num_steps=1):
        """Propogate input through the network."""

        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        output = []
        input_projection = []
        steps = range(input.size(0))
        for i in steps:
            for _ in range(num_steps):
                hidden = self.recurrence(input[i], hidden)
            output.append(hidden)
            input_projection.append(self.input2h(input[i]))

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        input_projection = torch.stack(input_projection, dim=0)

        return output, hidden, input_projection


class RNNNet(nn.Module):
    """Recurrent network model.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size

    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, **kwargs)

        # Add an output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, num_steps=1):
        rnn_output, hidden, input_projection = self.rnn(x, num_steps=num_steps)
        out = self.fc(rnn_output)
        return out, rnn_output


class TransformCTRNN(nn.Module):
    """Continuous-time RNN with learnable transformation matrix.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms. 
            If None, dt equals time constant tau
        train_transform_increment_coeff: If True, transform increment coefficient is learnable

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()

    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, train_transform_increment_coeff=False, train_alpha=False, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1.0
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        if train_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(
                alpha, dtype=torch.float32))

        if train_transform_increment_coeff:
            self.transform_increment_coeff = nn.Parameter(
                torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('transform_increment_coeff',
                                 torch.tensor(alpha, dtype=torch.float32))

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

        # Learnable transformation matrix
        self.transform = nn.Parameter(torch.randn(hidden_size**2, hidden_size))

        # Initialize transform_hidden as identity matrix
        self.register_buffer('transform_hidden', torch.eye(hidden_size))

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden, transform_hidden):
        """Run network for one time step."""
        input_proj = self.input2h(input)

        # Apply transformation
        transform_matrix = self.transform.matmul(
            input_proj.unsqueeze(-1)).view(-1, self.hidden_size, self.hidden_size)

        # Update transform_hidden
        transform_hidden = transform_hidden * \
            (1 - self.transform_increment_coeff) + \
            transform_matrix * self.transform_increment_coeff

        # Apply transformed input
        transformed_input = transform_hidden.bmm(
            input_proj.unsqueeze(-1)).squeeze(-1)

        h_new = torch.relu(transformed_input + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new, transform_hidden

    def forward(self, input, hidden=None, num_steps=1):
        """Propagate input through the network."""

        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Initialize transform_hidden for each item in the batch
        transform_hidden = self.transform_hidden.unsqueeze(
            0).repeat(input.shape[1], 1, 1)

        # Loop through time
        output = []
        steps = range(input.size(0))
        for i in steps:
            for _ in range(num_steps):
                hidden, transform_hidden = self.recurrence(
                    input[i], hidden, transform_hidden)
            output.append(hidden)

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)

        return output, hidden


class TransformRNNNet(nn.Module):
    """Recurrent network model with transformation matrix.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size

    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN with transformation
        self.rnn = TransformCTRNN(input_size, hidden_size, **kwargs)

        # Add an output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, num_steps=1):
        rnn_output, hidden = self.rnn(x, num_steps=num_steps)
        out = self.fc(rnn_output)
        return out, rnn_output


class TransformCTRNN2(nn.Module):
    """Continuous-time RNN with learnable transformation matrix based on input and hidden state.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms. 
            If None, dt equals time constant tau
        train_transform_increment_coeff: If True, transform increment coefficient is learnable

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()

    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, train_transform_increment_coeff=False, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1.0
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        if train_transform_increment_coeff:
            self.transform_increment_coeff = nn.Parameter(
                torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('transform_increment_coeff',
                                 torch.tensor(alpha, dtype=torch.float32))

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

        # Learnable transformation matrix
        self.transform = nn.Parameter(
            torch.randn(hidden_size**2, 2*hidden_size))

        # Initialize transform_hidden as identity matrix
        self.register_buffer('transform_hidden', torch.eye(hidden_size))

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden, transform_hidden):
        """Run network for one time step."""
        input_proj = self.input2h(input)

        # Concatenate input projection and hidden state
        combined = torch.cat([input_proj, hidden], dim=1)

        # Apply transformation
        transform_matrix = self.transform.matmul(
            combined.unsqueeze(-1)).view(-1, self.hidden_size, self.hidden_size)

        # Update transform_hidden
        transform_hidden = transform_hidden * \
            (1 - self.transform_increment_coeff) + \
            transform_matrix * self.transform_increment_coeff

        # Apply transformed input
        transformed_input = transform_hidden.bmm(
            input_proj.unsqueeze(-1)).squeeze(-1)

        h_new = torch.relu(transformed_input + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new, transform_hidden

    def forward(self, input, hidden=None, num_steps=1):
        """Propagate input through the network."""

        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Initialize transform_hidden for each item in the batch
        transform_hidden = self.transform_hidden.unsqueeze(
            0).repeat(input.shape[1], 1, 1)

        # Loop through time
        output = []
        steps = range(input.size(0))
        for i in steps:
            for _ in range(num_steps):
                hidden, transform_hidden = self.recurrence(
                    input[i], hidden, transform_hidden)
            output.append(hidden)

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)

        return output, hidden


class TransformRNNNet2(nn.Module):
    """Recurrent network model with transformation matrix based on input and hidden state.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size

    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN with transformation
        self.rnn = TransformCTRNN2(input_size, hidden_size, **kwargs)

        # Add an output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, num_steps=1):
        rnn_output, hidden = self.rnn(x, num_steps=num_steps)
        out = self.fc(rnn_output)
        return out, rnn_output


class ContextTransformCTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dt=None, train_alpha=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1.0
        else:
            alpha = dt / self.tau

        if train_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(
                alpha, dtype=torch.float32))

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

        # Context RNN
        self.input2context = nn.Linear(input_size, hidden_size)
        self.context2context = nn.Linear(hidden_size, hidden_size)

        # Transform computation
        self.context2transform = nn.Linear(hidden_size, hidden_size ** 2)

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.hidden_size))

    def recurrence(self, input, hidden, context):
        # Apply transformed input
        input_proj = self.input2h(input)

        # Compute transform matrix from context
        transform_matrix = self.context2transform(
            context).view(-1, self.hidden_size, self.hidden_size)

        # Apply transform_matrix to input_proj
        transformed_input = torch.bmm(
            transform_matrix, input_proj.unsqueeze(-1)).squeeze(-1)

        # Update hidden state
        h_new = torch.relu(context + transformed_input + self.h2h(hidden))
        hidden = hidden * (1 - self.alpha) + h_new * self.alpha

        # Update context
        context_input_proj = self.input2context(input)
        context_new = torch.relu(self.context2context(
            context) + context_input_proj)
        context = context * (1 - self.alpha) + context_new * self.alpha

        hidden = transformed_input

        return hidden, context

    def forward(self, input, hidden=None, num_steps=1):
        if hidden is None:
            hidden, context = self.init_hidden(input.shape[1])
            hidden = hidden.to(input.device)
            context = context.to(input.device)
        else:
            hidden, context = hidden

        output = []
        steps = range(input.size(0))
        for i in steps:
            for _ in range(num_steps):
                hidden, context = self.recurrence(
                    input[i], hidden, context)
            # output.append(torch.cat([hidden, context], dim=1))
            output.append(hidden)

        output = torch.stack(output, dim=0)
        return output, (hidden, context)


class ContextTransformRNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.rnn = ContextTransformCTRNN(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None, num_steps=1):
        rnn_output, hidden = self.rnn(x, hidden, num_steps=num_steps)
        out = self.fc(rnn_output)
        return out, rnn_output


class ContextCTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dt=None, train_transform_increment_coeff=False, train_alpha=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1.0
        else:
            alpha = dt / self.tau

        if train_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(
                alpha, dtype=torch.float32))

        if train_transform_increment_coeff:
            self.transform_increment_coeff = nn.Parameter(
                torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('transform_increment_coeff',
                                 torch.tensor(alpha, dtype=torch.float32))

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

        # Context RNN
        self.context2context = nn.Linear(hidden_size, hidden_size)
        self.input2context = nn.Linear(input_size, hidden_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.hidden_size))

    def recurrence(self, input, hidden, context):
        # Apply transformed input
        input_proj = self.input2h(input) + context

        # Update hidden state
        h_new = torch.relu(input_proj + self.h2h(hidden))
        hidden = hidden * (1 - self.alpha) + h_new * self.alpha

        # Update context
        context_new = torch.relu(self.context2context(
            context) + self.input2context(input))
        context = context * (1 - self.alpha) + context_new * self.alpha

        return hidden, context

    def forward(self, input, hidden=None, num_steps=1):
        if hidden is None:
            hidden, context = self.init_hidden(input.shape[1])
            hidden = hidden.to(input.device)
            context = context.to(input.device)
        else:
            hidden, context = hidden

        output = []
        steps = range(input.size(0))
        for i in steps:
            for _ in range(num_steps):
                hidden, context = self.recurrence(
                    input[i], hidden, context)
            # output.append(torch.cat([hidden, context], dim=1))
            output.append(hidden)

        output = torch.stack(output, dim=0)
        return output, (hidden, context)


class ContextRNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.rnn = ContextCTRNN(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None, num_steps=1):
        rnn_output, hidden = self.rnn(x, hidden, num_steps=num_steps)
        out = self.fc(rnn_output)
        return out, rnn_output


class ActionMapCTRNN(nn.Module):
    def __init__(self, input_size, context_size, output_size, use_tanh=False, dt=None, train_alpha=False):
        super().__init__()
        self.input_size = input_size
        self.context_size = context_size
        self.output_size = output_size
        self.tau = 100
        if dt is None:
            alpha = 1.0
        else:
            alpha = dt / self.tau

        if train_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(
                alpha, dtype=torch.float32))

        # Context RNN
        self.context2context = nn.Linear(self.context_size, self.context_size)
        self.input2context = nn.Linear(input_size, self.context_size)

        self.context2action_map = nn.Linear(
            self.context_size, output_size * input_size)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.context_size)

    def recurrence(self, input, context):
        # Compute action map from context
        action_map = self.context2action_map(
            context).view(-1, self.output_size, self.input_size)

        # Update context
        context_new = torch.relu(self.context2context(
            context) + self.input2context(input))
        context = context * (1 - self.alpha) + context_new * self.alpha

        # Apply action map to input
        output = torch.bmm(action_map, input.unsqueeze(-1)).squeeze(-1)

        return output, context

    def forward(self, input, context=None, num_steps=1):
        if context is None:
            context = self.init_hidden(input.shape[1])
            context = context.to(input.device)
        else:
            context = context

        outputs = []
        steps = range(input.size(0))
        for i in steps:
            output = None
            for _ in range(num_steps):
                output, context = self.recurrence(
                    input[i], context)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        return outputs, context


class ActionMapRNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.rnn = ActionMapCTRNN(
            input_size, hidden_size, output_size, **kwargs)

    def forward(self, x, hidden=None, num_steps=1):
        return self.rnn(x, hidden, num_steps=num_steps)


class BasicResidualNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        # out = self.activation(out)
        # out = self.fc2(out)
        # out = self.activation(out)
        return out


class OutputMapCTRNN(nn.Module):
    def __init__(self, input_size, context_size, hidden_size, output_size, use_tanh=False, dt=None, train_alpha=False):
        super().__init__()
        self.input_size = input_size
        self.context_size = context_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1.0
        else:
            alpha = dt / self.tau

        if train_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(
                alpha, dtype=torch.float32))

        # Context RNN
        self.context2context = nn.Linear(self.context_size, self.context_size)
        self.input2context = nn.Linear(input_size, self.context_size)

        self.input2hidden = BasicResidualNN(input_size, self.hidden_size)

        self.hidden2hidden = nn.Linear(self.hidden_size, self.hidden_size)

        self.context2output_map = nn.Linear(
            self.context_size, self.hidden_size * output_size)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.context_size)

    def recurrence(self, input, context):
        # Compute action map from context
        output_map = self.context2output_map(
            context).view(-1, self.output_size, self.hidden_size)
        # add layer norm
        output_map = nn.LayerNorm(self.hidden_size)(output_map)

        # Update context
        context_new = torch.relu(self.context2context(
            context) + self.input2context(input))
        context = context * (1 - self.alpha) + context_new * self.alpha

        hidden0 = torch.relu(self.input2hidden(input))
        hidden1 = torch.relu(self.hidden2hidden(hidden0)) + hidden0

        # Apply action map to input
        output = torch.bmm(output_map, hidden1.unsqueeze(-1)).squeeze(-1)

        return output, context

    def forward(self, input, context=None, num_steps=1):
        if context is None:
            context = self.init_hidden(input.shape[1])
            context = context.to(input.device)
        else:
            context = context

        outputs = []
        steps = range(input.size(0))
        for i in steps:
            output = None
            for _ in range(num_steps):
                output, context = self.recurrence(
                    input[i], context)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        return outputs, context


class OutputMapRNNNet(nn.Module):
    def __init__(self, input_size, context_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.rnn = OutputMapCTRNN(
            input_size, context_size, hidden_size, output_size, **kwargs)

    def forward(self, x, context=None, num_steps=1):
        return self.rnn(x, context, num_steps=num_steps)


class HiddenMapCTRNN(nn.Module):
    def __init__(self, input_size, context_size, hidden_size, output_size, use_tanh=False, dt=None, train_alpha=False):
        super().__init__()
        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.tau = 100
        if dt is None:
            alpha = 1.0
        else:
            alpha = dt / self.tau

        if train_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(
                alpha, dtype=torch.float32))

        # Context RNN
        self.context2context = nn.Linear(self.context_size, self.context_size)
        self.input2context = nn.Linear(input_size, self.context_size)

        # Input to hidden and hidden to output
        self.input2hidden = nn.Linear(input_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)

        # Context to hidden map
        self.context2hidden_map = nn.Linear(
            self.context_size, hidden_size * hidden_size)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Activation function
        self.activation = nn.Tanh() if use_tanh else nn.ReLU()

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.context_size)

    def recurrence(self, input, context):
        # Compute hidden map from context
        hidden_map = self.context2hidden_map(
            context).view(-1, self.hidden_size, self.hidden_size)
        hidden_map = self.layer_norm(hidden_map)

        # Update context
        context_new = self.activation(self.context2context(
            context) + self.input2context(input))
        context = context * (1 - self.alpha) + context_new * self.alpha

        # Compute hidden state
        hidden0 = self.activation(self.input2hidden(input))
        hidden1 = torch.bmm(hidden_map, hidden0.unsqueeze(-1)).squeeze(-1)
        hidden1 = self.activation(hidden1) + hidden0

        # Compute output
        output = self.hidden2output(hidden1)

        return output, context

    def forward(self, input, context=None, num_steps=1):
        if context is None:
            context = self.init_hidden(input.shape[1]).to(input.device)

        outputs = []
        steps = range(input.size(0))
        for i in steps:
            output = None
            for _ in range(num_steps):
                output, context = self.recurrence(input[i], context)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        return outputs, context


class HiddenMapRNNNet(nn.Module):
    def __init__(self, input_size, context_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.rnn = HiddenMapCTRNN(
            input_size, context_size, hidden_size, output_size, **kwargs)

    def forward(self, x, context=None, num_steps=1):
        return self.rnn(x, context, num_steps=num_steps)


class InputMapCTRNN(nn.Module):
    def __init__(self, input_size, context_size, hidden_size, output_size, use_tanh=False, dt=None, train_alpha=False):
        super().__init__()
        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.tau = 100
        if dt is None:
            alpha = 1.0
        else:
            alpha = dt / self.tau

        if train_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(
                alpha, dtype=torch.float32))

        # Context RNN
        self.context2context = nn.Linear(self.context_size, self.context_size)
        self.input2context = nn.Linear(input_size, self.context_size)

        # Input map
        self.context2input_map = nn.Linear(
            self.context_size, input_size * hidden_size)

        # Hidden to hidden and hidden to output
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)

        # Layer normalization
        self.layer_norm_hidden = nn.LayerNorm(hidden_size)

        # Activation function
        self.activation = nn.Tanh() if use_tanh else nn.ReLU()

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.context_size)

    def recurrence(self, input, context):
        # Compute input map from context
        input_map = self.context2input_map(
            context).view(-1, self.hidden_size, self.input_size)

        # Update context
        context_new = self.activation(self.context2context(
            context) + self.input2context(input))
        context = context * (1 - self.alpha) + context_new * self.alpha

        # Apply input map to input
        hidden = torch.bmm(input_map, input.unsqueeze(-1)).squeeze(-1)
        hidden = self.activation(hidden)

        # Process through hidden layer with skip connection
        hidden_new = self.hidden2hidden(hidden)
        hidden_new = self.layer_norm_hidden(hidden_new)
        hidden = hidden + self.activation(hidden_new)

        # Compute output
        output = self.hidden2output(hidden)

        return output, context

    def forward(self, input, context=None, num_steps=1):
        if context is None:
            context = self.init_hidden(input.shape[1]).to(input.device)

        outputs = []
        steps = range(input.size(0))
        for i in steps:
            output = None
            for _ in range(num_steps):
                output, context = self.recurrence(input[i], context)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        return outputs, context


class InputMapRNNNet(nn.Module):
    def __init__(self, input_size, context_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.rnn = InputMapCTRNN(
            input_size, context_size, hidden_size, output_size, **kwargs)

    def forward(self, x, context=None, num_steps=1):
        return self.rnn(x, context, num_steps=num_steps)
