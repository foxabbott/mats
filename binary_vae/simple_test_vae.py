import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import numpy as np
from pgmpy.factors.discrete import TabularCPD
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_gen import generate_continuous_variables, generate_continuous_variables_simple, generate_continuous_variables_super_simple, create_and_sample_bayesian_network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BinaryVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(BinaryVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, input_dim)
        
    def forward(self, x):
        latents = torch.sigmoid(self.fc1(x))
        x_recon = self.fc2(latents)
        return x_recon, latents

    def sample(self, z):
        return self.decode(z)

def loss_function(x_recon, x, latents):
    BCE = F.mse_loss(x_recon, x, reduction='sum')  # Reconstruction loss

   # Regularization term to encourage latents to be close to 0 or 1
    reg_term = torch.sum(latents * (1 - latents))
    return BCE + reg_term

def train_vae(model, data, epochs=100, batch_size=64, learning_rate=1e-3, clip_grads=False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (x,) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, latents = model(x)
            loss = loss_function(x_recon, x, latents)
            loss.backward()
            if clip_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.6f}')


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
    noise_std = 0
    continuous_var_dim = 10
    # continuous_generation_function = generate_continuous_variables
    # continuous_generation_function = generate_continuous_variables_simple
    continuous_generation_function = generate_continuous_variables_super_simple
    continuous_samples = continuous_generation_function(binary_samples[['H', 'I', 'J']], 
                                                       continuous_var_dim=continuous_var_dim, noise_std=noise_std)

    # Normalise data
    continuous_samples = continuous_samples / continuous_samples.std()

    continuous_data_tensor = torch.tensor(continuous_samples.values, dtype=torch.float32).to(device)
    input_dim = continuous_data_tensor.shape[1]
    # latent_dim = binary_samples.shape[1]

    latent_dim = 3

    model = BinaryVAE(input_dim, latent_dim).to(device)
    lr = 1e-2
    clip_grads = False

    train_vae(model, continuous_data_tensor, learning_rate=lr, clip_grads=clip_grads)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass to get reconstructed output and logits
    with torch.no_grad():
        x_recon, latents = model(continuous_data_tensor)
        x_recon_df = pd.DataFrame(x_recon)
        latents_df = pd.DataFrame(latents)

        corrs_binary = pd.concat([binary_samples, latents_df], axis=1).corr().loc[binary_samples.columns, latents_df.columns]
        corrs_cts = pd.concat([continuous_samples, x_recon_df], axis=1).corr().loc[continuous_samples.columns, x_recon_df.columns]
        import pdb; pdb.set_trace()

"""
Next steps:
    - Here we get perfect reconstruction, but the latents again don't really correspond to the binary vars
    - Indication this problem is non-identifiable: think about this tomorrow.
"""