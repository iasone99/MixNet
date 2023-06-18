import torch.nn as nn
import torch
import torch.nn.functional as F
import hyperparams as hp


class SliceDNN(nn.Module):

    def __init__(self, input_len, num_layers, hidden_size, output_len):
        super(SliceDNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_len = input_len
        self.output_len = output_len

        self.norm = nn.BatchNorm1d(hidden_size)
        self.linear_in = nn.Linear(input_len, hidden_size)
        self.linear_h = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ELU()
        # The linear layer that maps from hidden state space to mel
        self.linear_out = nn.Linear(hidden_size, output_len)

    def forward(self, mel):  # mel is of shape [B,T,F]
        mel1 = mel[:, :hp.num_frames, ]
        mel2 = mel[:, hp.num_frames:, :]
        mel = torch.cat((mel1, mel2), dim=2)
        mel_split = mel.split(1, dim=1)
        outputs = []
        for frame in mel_split:  # N, 1
            frame = frame.squeeze(1)
            x = self.linear_in(frame)
            x = self.norm(x)
            x = self.act(x)
            for i in range(self.num_layers):
                x = self.linear_h(x)
                x = self.norm(x)
                x = self.act(x)
            out = (self.linear_out(x))
            outputs.append(out.unsqueeze(1))

        # transform list to tensor
        output = torch.cat(outputs, dim=1)  # [B,T,F]

        return output
