import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import numpy as np
from pgmpy.factors.discrete import TabularCPD
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_gen import generate_continuous_variables, create_and_sample_bayesian_network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

class BinaryVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(BinaryVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mu = torch.sigmoid(self.fc_mu(h2))  # Sigmoid to ensure output is between 0 and 1
        return mu
    
    def reparameterize(self, mu):
        # Sample from Bernoulli distribution
        return Bernoulli(mu).sample()
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        x_recon = self.fc5(h4)
        return x_recon
    
    def forward(self, x):
        mu = self.encode(x)
        z = self.reparameterize(mu)
        x_recon = self.decode(z)
        return x_recon, mu

    def sample(self, z):
        return self.decode(z)

def loss_function(x_recon, x, mu):
    BCE = F.mse_loss(x_recon, x, reduction='sum')  # Reconstruction loss
    # KL divergence for Bernoulli latent variables
    KLD = torch.sum(mu * torch.log(mu / 0.5) + (1 - mu) * torch.log((1 - mu) / 0.5))
    return BCE + KLD

def train_vae(model, data, epochs=100, batch_size=64, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (x,) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu = model(x)
            loss = loss_function(x_recon, x, mu)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.6f}')

def inspect_latent_space(model, data):
    model.eval()
    with torch.no_grad():
        mu = model.encode(data)
        z = (mu > 0.5).float()  # Binarize the latent variables
        return z.cpu().numpy()

if __name__ == "__main__":
    # Define the structure of the Bayesian Network
    structure = [
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'D'),
        ('C', 'D'),
        ('D', 'E'),
        ('B', 'F'),
        ('C', 'G'),
        ('E', 'H'),
        ('F', 'H'),
        ('G', 'I'),
        ('H', 'J'),
    ]
    structure = []

    # Define the CPDs (Conditional Probability Distributions)
    cpds = {
        'A': TabularCPD(variable='A', variable_card=2, values=[[0.5], [0.5]]),
        'B': TabularCPD(variable='B', variable_card=2, values=[[0.8, 0.2], [0.2, 0.8]], evidence=['A'], evidence_card=[2]),
        'C': TabularCPD(variable='C', variable_card=2, values=[[0.7, 0.3], [0.3, 0.7]], evidence=['A'], evidence_card=[2]),
        'D': TabularCPD(variable='D', variable_card=2, values=[[0.9, 0.4, 0.6, 0.1], [0.1, 0.6, 0.4, 0.9]], evidence=['B', 'C'], evidence_card=[2, 2]),
        'E': TabularCPD(variable='E', variable_card=2, values=[[0.95, 0.5], [0.05, 0.5]], evidence=['D'], evidence_card=[2]),
        'F': TabularCPD(variable='F', variable_card=2, values=[[0.85, 0.3], [0.15, 0.7]], evidence=['B'], evidence_card=[2]),
        'G': TabularCPD(variable='G', variable_card=2, values=[[0.9, 0.4], [0.1, 0.6]], evidence=['C'], evidence_card=[2]),
        'H': TabularCPD(variable='H', variable_card=2, values=[[0.8, 0.5, 0.6, 0.3], [0.2, 0.5, 0.4, 0.7]], evidence=['E', 'F'], evidence_card=[2, 2]),
        'I': TabularCPD(variable='I', variable_card=2, values=[[0.7, 0.2], [0.3, 0.8]], evidence=['G'], evidence_card=[2]),
        'J': TabularCPD(variable='J', variable_card=2, values=[[0.9, 0.6], [0.1, 0.4]], evidence=['H'], evidence_card=[2]),
    }
    cpds = {
        'H': TabularCPD(variable='H', variable_card=2, values=[[0.5], [0.5]]),
        'I': TabularCPD(variable='I', variable_card=2, values=[[0.5], [0.5]]),
        'J': TabularCPD(variable='J', variable_card=2, values=[[0.5], [0.5]]),
    }

    # Sample from the Bayesian Network
    print("Building binary samples...")
    binary_samples = create_and_sample_bayesian_network(structure, cpds, sample_size=10000)

    # Generate continuous variables
    print("\nBuilding continuous samples...")
    continuous_samples = generate_continuous_variables(binary_samples[['H', 'I', 'J']], 
                                                       continuous_var_dim=20, noise_std=0.001)
    print(continuous_samples.head())

    continuous_data_tensor = torch.tensor(continuous_samples.values, dtype=torch.float32).to(device)
    input_dim = continuous_data_tensor.shape[1]
    latent_dim = binary_samples.shape[1]

    model = BinaryVAE(input_dim, latent_dim).to(device)
    train_vae(model, continuous_data_tensor)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass to get reconstructed output, mu, and logvar
    with torch.no_grad():
        x_recon, mu = model(continuous_data_tensor)

        latent_space = inspect_latent_space(model, continuous_data_tensor)
        latent_space_df = pd.DataFrame(latent_space)

        corrs = pd.concat([binary_samples, latent_space_df], axis=1).corr().loc[binary_samples.columns, latent_space_df.columns]
        import pdb; pdb.set_trace()
        """
        Notes:
            - Reconstruction is basically constant for all inputs. Why is this? 
            - Latents are almost always mapped to 0.5... they should hopefully map to a permutation of binary_samples
            - So no learning is taking place
        """
