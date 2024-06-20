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

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.fc(x)


def loss_function(yhat, y):
    BCE = F.mse_loss(yhat, y, reduction='sum')  # Reconstruction loss
    return BCE


def train_vae(model, x_data, y_data, epochs=100, batch_size=64, learning_rate=1e-3, clip_grads=False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for _, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            loss = loss_function(yhat, y)
            loss.backward()
            if clip_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.6f}')


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
    noise_std = 1e-5
    continuous_var_dim = 5
    continuous_generation_function = generate_continuous_variables_super_simple
    continuous_samples = continuous_generation_function(binary_samples[['H', 'I', 'J']], 
                                                       continuous_var_dim=continuous_var_dim, noise_std=noise_std)

    # Normalise data
    continuous_samples = continuous_samples / continuous_samples.std()

    binary_data_tensor = torch.tensor(binary_samples[['H', 'I', 'J']].values, dtype=torch.float32).to(device)
    continuous_data_tensor = torch.tensor(continuous_samples.values, dtype=torch.float32).to(device)

    input_dim = binary_data_tensor.shape[1]
    output_dim = continuous_data_tensor.shape[1]

    model = SimpleNN(input_dim, output_dim).to(device)
    lr = 1e-3
    clip_grads = False
    train_vae(model, binary_data_tensor, continuous_data_tensor, learning_rate=lr, clip_grads=clip_grads)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass to get reconstructed output
    with torch.no_grad():
        yhat = model(binary_data_tensor)
        yhat_df = pd.DataFrame(yhat.cpu().numpy())

        corrs_cts = pd.concat([continuous_samples, yhat_df], axis=1).corr().loc[continuous_samples.columns, yhat_df.columns]
        import pdb; pdb.set_trace()