import torch.nn as nn
import torch
import torch.nn.functional as F


class MixDNN(nn.Module):

    def __init__(self, input_len, hidden_size, num_layers, output_len, num_chunks_in, num_chunks_out):
        super(MixDNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_len = input_len
        self.num_chunks_in = num_chunks_in
        self.num_chunks_out = num_chunks_out
        self.output_len = output_len

        self.norm = nn.BatchNorm1d(hidden_size)
        self.linear_in = nn.Linear(input_len, hidden_size)
        self.linear_h = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()
        # The linear layer that maps from hidden state space to mel
        self.linear_out = nn.Linear(hidden_size, output_len)

    def forward(self, mel):  # mel is of shape [B,features,T]
        mel = torch.reshape(mel, (-1, self.input_len))
        # N, 1
        x = self.linear_in(mel)
        x = self.norm(x)
        x = self.act(x)
        for i in range(self.num_layers):
            x = self.linear_h(x)
            x = self.norm(x)
            x = self.act(x)
        out = self.linear_out(x)
        out = F.relu(out)  # [B, T*(2*T)]

        mask = torch.reshape(out, (-1, self.num_chunks_out, self.num_chunks_in))  # [B, num_chunks, T]
        return mask
