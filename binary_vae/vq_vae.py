import itertools
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import numpy as np
from pgmpy.factors.discrete import TabularCPD
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_gen import generate_continuous_variables, generate_continuous_variables_simple, generate_continuous_variables_super_simple, create_and_sample_bayesian_network
from abc import ABC, abstractmethod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BinaryVectorQuantizer(nn.Module):
    def __init__(self, num_latents):
        super(BinaryVectorQuantizer, self).__init__()
        self.num_latents = num_latents
        self.num_embeddings = 2**num_latents
        
        # Initialize codebook with all binary combinations
        binary_combinations = list(itertools.product([0, 1], repeat=num_latents))
        self.register_buffer('codebook', torch.tensor(binary_combinations, dtype=torch.float32))

    def forward(self, inputs):
        # Flatten input
        flat_input = inputs.view(-1, self.num_latents)
        
        # Calculate distances
        distances = torch.cdist(flat_input, self.codebook)
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook[encoding_indices]
        
        # Straight-through estimator
        quantized_st = flat_input + (quantized - flat_input).detach()
        
        # Reshape quantized to match input shape
        quantized_st = quantized_st.view(inputs.shape)
        
        # Compute loss
        commitment_loss = F.mse_loss(quantized_st.detach(), inputs)
        if random.random() < 0.01:
            print(len(pd.DataFrame(quantized.detach().numpy()).round(2).value_counts()))
        return quantized_st, commitment_loss, encoding_indices

class BinaryVQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_latents):
        super(BinaryVQVAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, num_latents)
        )
        
        self.vq_layer = BinaryVectorQuantizer(num_latents)
        
        self.decoder = nn.Sequential(
            nn.Linear(num_latents, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, commitment_loss, encoding_indices = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return x_recon, commitment_loss, encoding_indices

# Training loop
def train_binary_vqvae(model, data_loader, epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_commitment_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            
            x_recon, commitment_loss, _ = model(batch)
            recon_loss = F.mse_loss(x_recon, batch)
            loss = recon_loss + commitment_loss
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_commitment_loss += commitment_loss.item()
        

        avg_loss = total_loss / len(data_loader)
        avg_recon_loss = total_recon_loss / len(data_loader)
        avg_commitment_loss = total_commitment_loss / len(data_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f} (recon loss: {avg_recon_loss}, commitment loss: {avg_commitment_loss})')



def inspect_latent_space(model, data):
    model.eval()
    with torch.no_grad():
        probs = model.encoder(data)
        z = (probs > 0.5).float()  # Binarize the latent variables
        return z.cpu().numpy()


if __name__ == "__main__":
    # Define the structure of the Bayesian Network
    structure = []

    # Define the CPDs (Conditional Probability Distributions)
    binary_var_names = ['H', 'I', 'J']
    # binary_var_names = ['A', 'B', 'C', 'D', 'E']
    # binary_var_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    cpds = {name: TabularCPD(variable=name, variable_card=2, values=[[0.5], [0.5]]) for name in binary_var_names}
 
    # Sample from the Bayesian Network
    print("Building binary samples...")
    sample_size = 10000
    binary_samples = create_and_sample_bayesian_network(structure, cpds, sample_size=sample_size)
    binary_data_tensor = torch.tensor(binary_samples.values, dtype=torch.float32).to(device)

    # Generate continuous variables
    print("\nBuilding continuous samples...")
    noise_std = 0
    continuous_var_dim = 10
    # cts_data_func = generate_continuous_variables 
    # cts_data_func = generate_continuous_variables_simple
    cts_data_func = generate_continuous_variables_super_simple

    continuous_samples = cts_data_func(binary_samples[binary_var_names], 
                                                    continuous_var_dim=continuous_var_dim, 
                                                    noise_std=noise_std)
    
    # Normalise data
    continuous_samples = continuous_samples / continuous_samples.std()

    continuous_data_tensor = torch.tensor(continuous_samples.values, dtype=torch.float32).to(device)
    input_dim = continuous_data_tensor.shape[1]
    # latent_dim = binary_samples.shape[1]
    latent_dim = len(binary_var_names)
    model_type = BinaryVQVAE
    lr = 1e-6

    epochs = 200000
    hidden_dim = 2**7
    batch_size = 128

    model = model_type(input_dim, hidden_dim, latent_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_loader = DataLoader(continuous_data_tensor, batch_size=batch_size, shuffle=True)

    # Assuming you have your data in a DataLoader called 'data_loader'
    model = model_type(input_dim, hidden_dim, latent_dim)
    train_binary_vqvae(model, data_loader, epochs=epochs, learning_rate=lr)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass to get reconstructed output and logits
    with torch.no_grad():
        x_recon, probs, samples = model(continuous_data_tensor)
        x_recon_df = pd.DataFrame(x_recon)

        latent_space = inspect_latent_space(model, continuous_data_tensor)
        latent_space_df = pd.DataFrame(latent_space)

        corrs_binary = pd.concat([binary_samples, latent_space_df], axis=1).corr().loc[binary_samples.columns, latent_space_df.columns]
        props_binary = pd.DataFrame([[np.mean(binary_samples[col1] == latent_space_df[col2]) for col2 in latent_space_df.columns] for col1 in binary_samples.columns], index=binary_samples.columns, columns=latent_space_df.columns)
        corrs_cts = pd.concat([continuous_samples, x_recon_df], axis=1).corr().loc[continuous_samples.columns, x_recon_df.columns]

        unique_combos = pd.concat([binary_samples, latent_space_df], axis=1).astype('int').value_counts().reset_index().sort_values(binary_var_names)
        info_recovered = len(unique_combos[latent_space_df.columns].value_counts()) == 2 ** len(binary_var_names)
        print(f"Binary proportions: \n {props_binary}")
        print(f"Information in binary variables {'recovered' if info_recovered else 'not recovered'}.")
        print(f"Binaries {'' if np.all([(1.0 in props_binary[i].values) or ((0.0 in props_binary[i].values)) for i in props_binary.columns]) else 'not '}perfectly recovered.")
        import pdb; pdb.set_trace()
