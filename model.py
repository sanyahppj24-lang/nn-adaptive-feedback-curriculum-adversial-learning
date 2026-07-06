import torch
import torch.nn as nn


# ==========================================================
# Xavier Initialization
# ==========================================================

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ==========================================================
# Mix & Transform Network (Paper)
# ==========================================================

class ConvStack(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.Sequential(

            # Layer 1 : [4,1,2]
            nn.Conv1d(
                in_channels=1,
                out_channels=2,
                kernel_size=4,
                stride=1,
                padding=2
            ),
            nn.Sigmoid(),

            # Layer 2 : [2,2,4]
            nn.Conv1d(
                in_channels=2,
                out_channels=4,
                kernel_size=2,
                stride=2
            ),
            nn.Sigmoid(),

            # Layer 3 : [1,4,4]
            nn.Conv1d(
                in_channels=4,
                out_channels=4,
                kernel_size=1,
                stride=1
            ),
            nn.Sigmoid(),

            # Layer 4 : [1,4,1]
            nn.Conv1d(
                in_channels=4,
                out_channels=1,
                kernel_size=1,
                stride=1
            ),
            nn.Tanh()

        )

        self.apply(init_weights)

    def forward(self, x):

        x = self.layers(x)

        return x.squeeze(1)


# ==========================================================
# Alice
# ==========================================================

class Alice(nn.Module):

    def __init__(self, msg_len=64, key_len=64):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(
                msg_len + key_len,
                msg_len + key_len
            ),

            nn.Sigmoid()

        )

        self.conv = ConvStack()

        self.apply(init_weights)

    def forward(self, msg, key):

        x = torch.cat((msg, key), dim=1)

        x = self.fc(x)

        x = x.unsqueeze(1)

        return self.conv(x)


# ==========================================================
# Bob
# ==========================================================

class Bob(nn.Module):

    def __init__(self, msg_len=64, key_len=64):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(
                msg_len + key_len,
                msg_len + key_len
            ),

            nn.Sigmoid()

        )

        self.conv = ConvStack()

        self.apply(init_weights)

    def forward(self, cipher, key):

        x = torch.cat((cipher, key), dim=1)

        x = self.fc(x)

        x = x.unsqueeze(1)

        return self.conv(x)


# ==========================================================
# Eve
# ==========================================================

class Eve(nn.Module):

    def __init__(self, msg_len=64, key_len=64):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(
                msg_len,
                msg_len + key_len
            ),

            nn.Sigmoid()

        )

        self.conv = ConvStack()

        self.apply(init_weights)

    def forward(self, cipher):

        x = self.fc(cipher)

        x = x.unsqueeze(1)

        return self.conv(x)