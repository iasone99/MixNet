import torch.nn as nn
import torch
import torch.nn.functional as F
import hyperparams as hp
from MixDNN import create_chunks


class MixCNN2(nn.Module):

    def __init__(self, input_len, hidden_size, num_layers, output_len, num_chunks_in, num_chunks_out):
        super(MixCNN2, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_len = input_len
        self.num_chunks_in = num_chunks_in
        self.num_chunks_out = num_chunks_out
        self.output_len = output_len

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(2, 1))
        self.bn1 = nn.GroupNorm(1, 16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 1))
        self.bn2 = nn.GroupNorm(1, 32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 1))
        self.bn3 = nn.GroupNorm(1, 64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 1))
        self.bn4 = nn.GroupNorm(1, 128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1))
        self.bn5 = nn.GroupNorm(1, 256)

        self.flat = nn.Flatten()
        self.pooling22 = nn.MaxPool2d(2, 2)

        self.norm = nn.BatchNorm1d(hidden_size)
        self.linear_in = nn.Linear(1280, hidden_size)
        self.linear_h = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()
        # The linear layer that maps from hidden state space to mel
        self.linear_out = nn.Linear(hidden_size, output_len)
        self.dropout = nn.Dropout(0.25)

    def forward(self, mel):
        mel3 = mel.unsqueeze(1)  # [B,C,T,F]
        # mel1 = mel[:, :, :hp.num_frames, :]
        # mel2 = mel[:, :, hp.num_frames:, :]
        # mel3 = torch.cat((mel1, mel2), dim=1)
        # (B, in_c, T, F)
        x1 = self.pooling22(F.elu(self.bn1(self.conv1(mel3))))
        x2 = self.pooling22(F.elu(self.bn2(self.conv2(x1))))
        x3 = self.pooling22(F.elu(self.bn3(self.conv3(x2))))
        x4 = self.pooling22(F.elu(self.bn4(self.conv4(x3))))
        # x5 = self.pooling22(F.elu(self.bn5(self.conv5(x4))))

        # x5 = torch.reshape(x5, (-1, self.input_len))
        x5 = self.flat(x4)
        # N, 1
        x = self.linear_in(x5)
        x = self.norm(x)
        x = self.act(x)

        for i in range(self.num_layers):
            x = self.linear_h(x)
            x = self.norm(x)
            x = self.act(x)
            x = self.dropout(x)
        out = self.linear_out(x)
        out = F.relu(out)

        mask = torch.reshape(out, (-1, self.num_chunks_out, self.num_chunks_in))  # [B, num_chunks, T]

        return mask
