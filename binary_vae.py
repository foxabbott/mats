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
    def __init__(self, input_dim, latent_dim, temp=0.5):
        super(BinaryVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.temp = temp

        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        
        # Reparameterisation
        self.reparameterisation_func = self.reparameterize_gumbel

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, input_dim)

        self.fc_decoder_simple = nn.Linear(latent_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        logits = torch.sigmoid(self.fc_mu(h2))  # Sigmoid to ensure output is between 0 and 1
        return logits
    
    def reparameterize_bernoulli(self, mu):
        # Sample from Bernoulli distribution
        return Bernoulli(mu).sample()
    
    def reparameterize_gumbel(self, logits):
        # Sample from Gumbel-Softmax distribution
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        y = (logits + gumbel_noise) / self.temp
        return F.softmax(y, dim=-1)

    # def decode(self, z):
    #     h3 = F.relu(self.fc3(z))
    #     h4 = F.relu(self.fc4(h3))
    #     x_recon = self.fc5(h4)
    #     return x_recon

    def decode(self, z):
        x_recon = self.fc_decoder_simple(z)
        return x_recon
    
    def forward(self, x):
        logits = self.encode(x)
        z = self.reparameterisation_func(logits)
        x_recon = self.decode(z)
        return x_recon, logits

    def sample(self, z):
        return self.decode(z)

def loss_function(x_recon, x, logits, beta=1.0):
    BCE = F.mse_loss(x_recon, x, reduction='sum')  # Reconstruction loss
    # KL divergence for Bernoulli latent variables
    # KLD = torch.sum(mu * torch.log(mu / 0.5) + (1 - mu) * torch.log((1 - mu) / 0.5))
    # print(f"BCE: {BCE}, KLD: {KLD}")

    # KL divergence for Gumbel-Softmax latent variables
    q_y = F.softmax(logits, dim=-1)
    log_q_y = F.log_softmax(logits, dim=-1)
    KLD = torch.sum(q_y * (log_q_y - torch.log(torch.tensor(1.0 / logits.size(-1)).to(device))), dim=-1).sum()
    
    return BCE + beta * KLD

def train_vae(model, data, epochs=100, batch_size=64, learning_rate=1e-3, beta=1.0, clip_grads=False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (x,) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, logits = model(x)
            loss = loss_function(x_recon, x, logits, beta)
            loss.backward()
            if clip_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.6f}')

def inspect_latent_space(model, data):
    model.eval()
    with torch.no_grad():
        logits = model.encode(data)
        z = (logits > 0.5).float()  # Binarize the latent variables
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
    noise_std = 1e-5
    continuous_var_dim = 5
    # continuous_generation_function = generate_continuous_variables
    # continuous_generation_function = generate_continuous_variables_simple
    continuous_generation_function = generate_continuous_variables_super_simple
    continuous_samples = continuous_generation_function(binary_samples[['H', 'I', 'J']], 
                                                       continuous_var_dim=continuous_var_dim, noise_std=noise_std)

    # Normalise data
    continuous_samples = continuous_samples / continuous_samples.std()

    continuous_data_tensor = torch.tensor(continuous_samples.values, dtype=torch.float32).to(device)
    input_dim = continuous_data_tensor.shape[1]
    latent_dim = binary_samples.shape[1]

    model = BinaryVAE(input_dim, latent_dim).to(device)
    lr = 1e-4
    beta = 0.1
    clip_grads = True
    train_vae(model, continuous_data_tensor, learning_rate=lr, beta=beta, clip_grads=clip_grads)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass to get reconstructed output and logits
    with torch.no_grad():
        x_recon, logits = model(continuous_data_tensor)
        x_recon_df = pd.DataFrame(x_recon)

        latent_space = inspect_latent_space(model, continuous_data_tensor)
        latent_space_df = pd.DataFrame(latent_space)

        corrs_binary = pd.concat([binary_samples, latent_space_df], axis=1).corr().loc[binary_samples.columns, latent_space_df.columns]
        corrs_cts = pd.concat([continuous_samples, x_recon_df], axis=1).corr().loc[continuous_samples.columns, x_recon_df.columns]
        import pdb; pdb.set_trace()
