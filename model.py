import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM(nn.Module):
    def __init__(self, config):
        super(CNN_LSTM, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=config['num_features'], out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(config['cnn_dropout_rate'])

        # LSTM layers
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.dropout_lstm = nn.Dropout(config['lstm_dropout_rate'])  # 0.5%

        # Dense layers
        self.fc1 = nn.Linear(128, 64)  # 128 is the hidden size of LSTM
        self.fc2 = nn.Linear(64, config['num_classes'])
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # He initialization for convolutional layers
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize bias to zero
            elif isinstance(m, nn.Linear):
                # Xavier initialization for fully connected layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Initialize bias to zero
            elif isinstance(m, nn.LSTM):
                # Initialize LSTM weights
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_uniform_(param, nonlinearity='linear')  # Input weights
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)  # Hidden weights
                    elif 'bias' in name:
                        nn.init.zeros_(param)  # Biases

    def forward(self, x):
        # CNN forward pass
        # print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout_cnn(x)

        # Preparing for LSTM (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1)  # Change to (batch_size, seq_len, features)

        # LSTM forward pass
        x, _ = self.lstm(x)
        x = self.dropout_lstm(x)

        # Use the output from the last time step
        x = x[:, -1, :]  # Get the last time step output

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x  # Output probabilities
