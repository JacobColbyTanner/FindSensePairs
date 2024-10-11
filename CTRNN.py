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

        return output, hidden


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
        rnn_output, hidden = self.rnn(x, num_steps=num_steps)
        out = self.fc(rnn_output)
        return out, hidden


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
    def __init__(self, input_size, context_size, hidden_size, output_size, use_tanh=False, dt=None, train_alpha=True):
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
        # Update context
        # context_new = torch.relu(self.context2context(
        #    context) + self.input2context(input))
        # context = context * (1 - self.alpha) + context_new * self.alpha

        # Here's a new idea for a recurrent network:
        # Maybe we can assume that at every point in time, there are
        # two learnable complementary subspaces of the embedding space, one for the context and one for the input,
        # but these subspaces change depending on the previous context.
        # So we can generate two matrices with complementary null spaces?
        # The context matrix maps any embedding vector with only input information to zero
        # and the input matrix maps any embedding vector with only context information to zero.
        # But what is input information and what is context information depends on the previous context.

        # Old version
        context = context * (1 - self.alpha) + \
            self.input2context(input) * self.alpha

        # Compute action map from context
        output_map = self.context2output_map(
            context).view(-1, self.output_size, self.hidden_size)
        # add layer norm
        output_map = nn.LayerNorm(self.hidden_size)(output_map)

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

    def forward(self, x, num_steps=1):
        return self.rnn(x, num_steps=num_steps)


class FeedFoward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, embedding_size, memory_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, memory_size),
            nn.Dropout(0),
        )

    def forward(self, x):
        return self.net(x)


class ContextInputSubspaceCTRNN(nn.Module):
    def __init__(self, input_size, memory_size, embedding_size, output_size, use_tanh=False, dt=None, train_alpha=True):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau

        if train_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(
                alpha, dtype=torch.float32))

        self.beta_power = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta_mult = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.input2embedding = nn.Linear(
            input_size, self.embedding_size, bias=False)

        self.embedding2memory = nn.Linear(
            self.embedding_size, self.memory_size, bias=False)
        self.embedding2actionable = nn.Linear(
            self.embedding_size, self.memory_size, bias=False)

        self.context2context_map = nn.Linear(
            self.memory_size, self.memory_size * self.output_size, bias=False)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.memory_size)

    def recurrence(self, input, context):

        input_embedding = self.input2embedding(input)

        context_portion = self.embedding2memory(input_embedding)
        actionable_portion = self.embedding2actionable(input_embedding)

        context_portion_norm = torch.norm(
            context_portion, p=2, dim=1).unsqueeze(1)
        context_norm = torch.norm(context, p=2, dim=1).unsqueeze(1)

        beta = self.alpha * (self.beta_mult * (context_portion_norm /
                             (context_portion_norm + context_norm))) ** self.beta_power

        # clamp beta to be between 0 and 1
        beta = torch.clamp(beta, 0, 1)

        context = (1-beta) * context + beta * context_portion

        # Compute action map from context
        context_map = self.context2context_map(
            context).view(-1, self.output_size, self.memory_size)

        # Apply action map to input
        output = torch.bmm(
            context_map, actionable_portion.unsqueeze(-1)).squeeze(-1)

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


class ContextInputSubspaceRNNNet(nn.Module):
    def __init__(self, input_size, memory_size, embedding_size, output_size, **kwargs):
        super().__init__()
        self.rnn = ContextInputSubspaceCTRNN(
            input_size, memory_size, embedding_size, output_size, **kwargs)

    def forward(self, x, num_steps=1):
        return self.rnn(x, num_steps=num_steps)


class ActionEmbeddingActionMap(nn.Module):
    def __init__(self, input_size, action_embedding_size, output_size, use_tanh=False, dt=None, train_alpha=True):
        super().__init__()
        self.input_size = input_size
        self.action_embedding_size = action_embedding_size
        self.output_size = output_size
        self.tau = 100
        if dt is None:
            alpha = 0.1
        else:
            alpha = dt / self.tau

        if train_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.tensor(
                alpha, dtype=torch.float32))

        # learn the initial action_map
        self.action_map_init = nn.Parameter(torch.randn(
            self.action_embedding_size ** 2))

        self.input2action_embedding = nn.Linear(
            input_size, self.action_embedding_size)

        self.action_embedding2action_map = nn.Linear(
            self.action_embedding_size, self.action_embedding_size ** 2, bias=False)

        self.action_embedding2action = nn.Linear(
            self.action_embedding_size, self.output_size)

        self.action_embedding_norm = nn.LayerNorm(self.action_embedding_size)
        self.action_map_norm = nn.LayerNorm(self.action_embedding_size ** 2)

    def init_hidden(self, batch_size):
        return self.action_map_init.repeat(batch_size, 1)

    def recurrence(self, input, action_map):

        action_embedding = self.input2action_embedding(input)

        # reshape action_map to be a square matrix
        action_map = action_map.view(-1, self.action_embedding_size,
                                     self.action_embedding_size)
        action_embedding = torch.bmm(
            action_map, action_embedding.unsqueeze(-1)).squeeze(-1)

        action_map = (1-self.alpha) * action_map.flatten(1) + \
            self.alpha * self.action_embedding2action_map(action_embedding)

        action_map = self.action_map_norm(action_map)
        action_embedding = self.action_embedding_norm(action_embedding)

        output = self.action_embedding2action(action_embedding)

        return output, action_map

    def forward(self, input, action_map=None, num_steps=1):
        if action_map is None:
            action_map = self.init_hidden(input.shape[1])
            action_map = action_map.to(input.device)
        else:
            action_map = action_map

        outputs = []
        steps = range(input.size(0))
        for i in steps:
            output = None
            for _ in range(num_steps):
                output, action_map = self.recurrence(
                    input[i], action_map)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        return outputs, action_map


class ActionEmbeddingActionMapNet(nn.Module):
    def __init__(self, input_size, action_embedding_size, output_size, **kwargs):
        super().__init__()
        self.rnn = ActionEmbeddingActionMap(
            input_size, action_embedding_size, output_size, **kwargs)

    def forward(self, x, num_steps=1):
        return self.rnn(x, num_steps=num_steps)


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


class OutputMapLSTM(nn.Module):
    def __init__(self, input_size, context_size, hidden_size, output_size, use_tanh=False):
        super().__init__()
        self.input_size = input_size
        self.context_size = context_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Change LSTM initialization to include num_layers
        self.num_layers = 1  # You can adjust this if needed
        self.lstm = nn.LSTM(input_size, context_size,
                            num_layers=self.num_layers)

        self.input2hidden = BasicResidualNN(input_size, self.hidden_size)

        self.hidden2hidden = nn.Linear(self.hidden_size, self.hidden_size)

        self.context2output_map = nn.Linear(
            self.context_size, self.hidden_size * output_size)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        # Activation function
        self.activation = nn.Tanh() if use_tanh else nn.ReLU()

    def init_hidden(self, batch_size):
        # Update to return 3D tensors
        return (torch.zeros(self.num_layers, batch_size, self.context_size),
                torch.zeros(self.num_layers, batch_size, self.context_size))

    def recurrence(self, input, hidden):
        # Unpack hidden state
        context, cell_state = hidden

        # Update context using LSTM
        context_input = input.unsqueeze(0)  # Add sequence length dimension
        context_output, (context, cell_state) = self.lstm(
            context_input, (context, cell_state))

        # Remove sequence length dimension from context_output
        context_output = context_output.squeeze(0)

        # Compute output map from context_output
        output_map = self.context2output_map(
            context_output).view(-1, self.output_size, self.hidden_size)
        output_map = self.layer_norm(output_map)

        hidden0 = self.activation(self.input2hidden(input))
        hidden1 = self.activation(self.hidden2hidden(hidden0)) + hidden0

        # Apply output map to hidden state
        output = torch.bmm(output_map, hidden1.unsqueeze(-1)).squeeze(-1)

        return output, (context, cell_state)

    def forward(self, input, hidden=None, num_steps=1):
        if hidden is None:
            hidden = self.init_hidden(input.shape[1])
            hidden = (hidden[0].to(input.device), hidden[1].to(input.device))

        outputs = []
        steps = range(input.size(0))
        for i in steps:
            output = None
            for _ in range(num_steps):
                output, hidden = self.recurrence(input[i], hidden)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        return outputs, hidden


class OutputMapLSTMNet(nn.Module):
    def __init__(self, input_size, context_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.rnn = OutputMapLSTM(
            input_size, context_size, hidden_size, output_size, **kwargs)

    def forward(self, x, hidden=None, num_steps=1):
        return self.rnn(x, hidden, num_steps=num_steps)


class OutputMapLSTMNet(nn.Module):
    def __init__(self, input_size, context_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.rnn = OutputMapLSTM(
            input_size, context_size, hidden_size, output_size, **kwargs)

    def forward(self, x, hidden=None, num_steps=1):
        return self.rnn(x, hidden, num_steps=num_steps)


class OutputMapContextMapCTRNN(nn.Module):
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

        self.context2hidden = nn.Linear(self.context_size, self.context_size)
        self.hidden2hidden = nn.Linear(self.context_size, self.context_size)

        self.context2output_map = nn.Linear(
            self.context_size, self.context_size * self.context_size)

        self.input2context_map = nn.Linear(
            self.context_size, self.context_size ** 2)

        self.output_layer = nn.Linear(self.context_size, output_size)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.context_size)

    def recurrence(self, input, context):
        input_proj = self.input2context(input)

        # Update context
        context_new = torch.relu(self.context2context(
            context) + input_proj)
        context = context * (1 - self.alpha) + context_new * self.alpha

        # Compute context map from input
        # context_map = self.input2context_map(
        #    input_proj).view(-1, self.context_size, self.context_size)
        # context_map = nn.LayerNorm(self.context_size)(context_map)

        # context = torch.bmm(context_map, context.unsqueeze(-1)
        #                    ).squeeze(-1) + input_proj

        hidden0 = torch.relu(self.context2hidden(context))
        hidden1 = torch.relu(self.hidden2hidden(hidden0)) + hidden0

        # Compute output map from context
        output_map = self.context2output_map(
            context).view(-1, self.context_size, self.context_size)
        # add layer norm
        output_map = nn.LayerNorm(self.context_size)(output_map)

        # Apply action map to input
        almost_output = torch.bmm(output_map, hidden1.unsqueeze(-1)
                                  ).squeeze(-1) + input_proj

        output = self.output_layer(almost_output)

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


class OutputMapContextMapRNNNet(nn.Module):
    def __init__(self, input_size, context_size, output_size, **kwargs):
        super().__init__()
        self.rnn = OutputMapContextMapCTRNN(
            input_size, context_size, output_size, **kwargs)

    def forward(self, x, num_steps=1):
        return self.rnn(x, num_steps=num_steps)


class SimpleLSTM(nn.Module):
    """Simple LSTM model with input and output projections.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        output_size: Number of output neurons

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tuple of tensors (h0, c0), each of shape (1, batch, hidden_size)
            if None, hidden is initialized through self.init_hidden()

    Outputs:
        output: tensor of shape (seq_len, batch, output_size)
        hidden: tuple of tensors (hn, cn), each of shape (1, batch, hidden_size)
        input_projection: tensor of shape (seq_len, batch, hidden_size)
    """

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input_projection = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=False)
        self.output_projection = nn.Linear(hidden_size, output_size)

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

    def forward(self, input, hidden=None, num_steps=1):
        """Propagate input through the network."""

        # If hidden state is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape)
            hidden = (hidden[0].to(input.device), hidden[1].to(input.device))

        # Project input
        input_projected = self.input_projection(input)

        # Process input through LSTM
        lstm_output, (hn, cn) = self.lstm(input_projected, hidden)

        # Project output
        output = self.output_projection(lstm_output)

        return output, cn
