import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Conditional Generator
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, binary_dim, continuous_dim, ordinal_dim, initial_temperature=0.2):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.binary_dim = binary_dim
        self.continuous_dim = continuous_dim
        self.ordinal_dim = ordinal_dim
        self.temperature = initial_temperature
        self.model = nn.Sequential(
            nn.Linear(latent_dim + binary_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, binary_dim + continuous_dim + ordinal_dim)
        )

    def forward(self, z, conditions):
        input = torch.cat([z, conditions], dim=1)
        logits = self.model(input)
        binary_logits = logits[:, :self.binary_dim]
        continuous_logits = logits[:, self.binary_dim:self.binary_dim + self.continuous_dim]
        ordinal_logits = logits[:, self.binary_dim + self.continuous_dim:]
        binary_output = torch.sigmoid(binary_logits / self.temperature)
        continuous_output = torch.sigmoid(continuous_logits)
        ordinal_output = torch.sigmoid(ordinal_logits)
        return torch.cat([binary_output, continuous_output, ordinal_output], dim=1)

    def update_temperature(self, new_temperature):
        self.temperature = new_temperature

class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, binary_dim):
        super(ConditionalDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + binary_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x, conditions):
        input = torch.cat([x, conditions], dim=1)
        return self.model(input)