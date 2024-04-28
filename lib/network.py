import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary

# Example Base encoder
class Encoder_1(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.description = "Encoder_1 - Base CNN Encoder"
        self.conv1_encoder = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=7, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 128, kernel_size=5, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 256, kernel_size=5, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 512, kernel_size=3, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 1024, kernel_size=3, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(1024, 1024, kernel_size=3, stride=1),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Flatten())

        self.linear_t_encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

        self.linear_q_encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )


# VO Encoder
class VONet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.cnn_encoder = Encoder_1(input_channels).conv1_encoder

        self.linear_t_encoder = Encoder_1(input_channels).linear_t_encoder

        self.linear_q_encoder = Encoder_1(input_channels).linear_q_encoder

    def forward(self, x):
        #print("x shape input", x.shape)
        x = self.cnn_encoder(x)
        #print("x shape", x.shape)
        t = self.linear_t_encoder(x)
        q = self.linear_q_encoder(x)
        return q,t


# IO Encoder
class IONet1(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer Call
        self.lstm = nn.LSTM(input_channels, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.linear_t_encoder = nn.Sequential(
            nn.Linear(hidden_size * 100, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

        self.linear_q_encoder = nn.Sequential(
            nn.Linear(hidden_size * 100, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)

        # Fully Connected
        t = self.linear_t_encoder(out)
        q = self.linear_q_encoder(out)

        return  q,t
    
class IONet2(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM Layer
        self.bilstm = nn.LSTM(input_channels, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Unidirectional LSTM Layer
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.linear_t_encoder = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

        self.linear_q_encoder = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        # Set initial hidden and cell states for bidirectional LSTM
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate bidirectional LSTM
        out, _ = self.bilstm(x, (h0, c0))

        # Forward propagate unidirectional LSTM
        out, _ = self.lstm(out)

        # Fully Connected
        t = self.linear_t_encoder(out[:, -1, :])  # Take the last time-step output
        q = self.linear_q_encoder(out[:, -1, :])  # Take the last time-step output

        return q, t


class IONet(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM Layer
        self.bilstm = nn.LSTM(input_channels, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Unidirectional LSTM Layer
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.linear_t_encoder = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

        self.linear_q_encoder = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        # Set initial hidden and cell states for bidirectional LSTM
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate bidirectional LSTM
        out, _ = self.bilstm(x, (h0, c0))

        # Forward propagate unidirectional LSTM
        out, _ = self.lstm(out)

        # Fully Connected
        t = self.linear_t_encoder(out[:, -1, :])  # Take the last time-step output
        q = self.linear_q_encoder(out[:, -1, :])  # Take the last time-step output

        return q, t

class VIONet(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers, num_classes):
        super().__init__()

        self.cnn_encoder = Encoder_1(input_channels).conv1_encoder

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM Layer
        self.bilstm = nn.LSTM(input_channels, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Unidirectional LSTM Layer
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True)

        self.linear_lstm_t_encoder = nn.Sequential(
            nn.Linear(hidden_size, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
        )

        self.linear_cnn_encoder = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
        )

        self.linear_t_encoder = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3)
        )

        self.linear_q_encoder = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4)
        )

    def forward(self, x, y):
        # Set initial hidden and cell states for bidirectional LSTM
        h0 = torch.zeros(self.num_layers*2, y.size(0), self.hidden_size).to(y.device)
        c0 = torch.zeros(self.num_layers*2, y.size(0), self.hidden_size).to(y.device)
        x = self.cnn_encoder(x)
        y, _ = self.bilstm(y, (h0, c0))
        y, _ = self.lstm(y)


        x = self.linear_cnn_encoder(x)
        y = self.linear_lstm_t_encoder(y[:, -1, :])
        z = torch.cat((x, y), dim=1)


        t = self.linear_t_encoder(z)
        q = self.linear_q_encoder(z)

        return q,t 
