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

        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, input_dim)
        
        # Reparameterisation
        self.reparameterisation_func = self.reparameterize_gumbel

        
    def encode(self, x):
        logits = self.fc1(x)
        return logits
    
    def reparameterize_gumbel(self, logits):
        # Sample from Gumbel-Softmax distribution
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        y = (logits + gumbel_noise) / self.temp
        return F.softmax(y, dim=-1)

    def sample_latent(self, logits):
        # Use the straight-through estimator
        probs = torch.sigmoid(logits)
        # Sample from Bernoulli
        z = torch.bernoulli(probs)
        # Apply straight-through estimator
        z_straight_through = z + (probs - probs.detach())
        return z_straight_through

    def decode(self, z):
        x_recon = self.fc2(z)
        return x_recon
    
    def forward(self, x):
        logits = self.encode(x)
        # z = self.reparameterisation_func(logits)
        z = self.sample_latent(logits)
        x_recon = self.decode(z)
        return x_recon, logits

    def sample(self, z):
        return self.decode(z)

def loss_function(x_recon, x, logits, beta=1.0):
    BCE = F.mse_loss(x_recon, x, reduction='sum')  # Reconstruction loss

    # KL divergence for Gumbel-Softmax latent variables
    # q_y = F.softmax(logits, dim=-1)
    # log_q_y = F.log_softmax(logits, dim=-1)
    # KLD = torch.sum(q_y * (log_q_y - torch.log(torch.tensor(1.0 / logits.size(-1)).to(device))), dim=-1).sum()

    # bernoulli KL divergence from binary prior
    q_y = torch.sigmoid(logits)
    # log_q_y = torch.log(q_y + 1e-10)  # Add small constant for numerical stability
    # log_1_minus_q_y = torch.log(1 - q_y + 1e-10)  # Add small constant for numerical stability

    # KLD = q_y * (log_q_y - torch.log(torch.tensor(0.5).to(device))) + \
    #       (1 - q_y) * (log_1_minus_q_y - torch.log(torch.tensor(0.5).to(device)))
    # import pdb; pdb.set_trace()
    # KLD = KLD.sum()  # Sum over all dimensions and batch

    # janky KLD (penalises close to 0.5)
    KLD = (q_y * (1 - q_y)).sum()
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
    structure = []

    # Define the CPDs (Conditional Probability Distributions)
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
    continuous_var_dim = 5
    continuous_samples = generate_continuous_variables_super_simple(binary_samples[['H', 'I', 'J']], 
                                                       continuous_var_dim=continuous_var_dim, 
                                                       noise_std=noise_std)

    # Normalise data
    continuous_samples = continuous_samples / continuous_samples.std()

    continuous_data_tensor = torch.tensor(continuous_samples.values, dtype=torch.float32).to(device)
    input_dim = continuous_data_tensor.shape[1]
    # latent_dim = binary_samples.shape[1]
    latent_dim = 3

    model = BinaryVAE(input_dim, latent_dim).to(device)
    lr = 1e-2
    beta = 0
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

"""
Next steps:
    - Is this actually an identifiable setup (with the super simple function)?
    - Is it identifiable if we make the function a bit more complex?
    - First try messing around with the parameters a bit
"""