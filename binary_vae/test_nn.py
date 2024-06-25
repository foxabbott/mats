import matplotlib.pyplot as plt
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
from abc import ABC, abstractmethod

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 2**7)
        self.fc2 = nn.Linear(2**7, output_dim)
        
    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        yhat = F.sigmoid(self.fc2(x1))
        return yhat


def loss_function(yhat, y, beta=0.0):
    BCE = F.mse_loss(yhat, y, reduction='sum')  # Reconstruction loss
    
    penalty = (yhat * (1 - yhat)).sum()
    return BCE + beta * penalty


def train_vae(model, x_data, y_data, epochs=100, batch_size=64, learning_rate=1e-3, clip_grads=False,
              beta=0.0, anneal_beta=False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        if anneal_beta:
            beta_epoch = beta * epoch / epochs
        else:
            beta_epoch = beta

        train_loss = 0
        for _, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            loss = loss_function(yhat, y, beta=beta_epoch)
            loss.backward()
            if clip_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_loss += loss.item()
            optimizer.step()


        # input_vals = np.arange(0, 4, 0.01).astype(np.float32)
        # encodings = pd.DataFrame(model(torch.tensor(input_vals).view(-1, 1)).detach().numpy())
        # encodings['input'] = input_vals
        # plt.figure(figsize=(10, 6))
        # for column in encodings.columns[:-1]:  # Exclude the 'input' column
        #     plt.plot(encodings['input'], encodings[column], label=f'Encoding {column}')
        # for value in [pd.DataFrame(x_data).value_counts().index[i][0] for i in range(len(pd.DataFrame(x_data).value_counts().index))]:
        #     plt.axvline(x=value, color='r', linestyle='--', linewidth=1)
        # plt.savefig(f"plots/simple_run/{epoch}.jpg")
        # plt.close()

        print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.6f}')


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

    # Generate continuous variables
    print("\nBuilding continuous samples...")
    noise_std = 1e-4
    continuous_var_dim = 1
    # cts_data_func = generate_continuous_variables 
    # cts_data_func = generate_continuous_variables_simple
    cts_data_func = generate_continuous_variables_super_simple

    continuous_samples = cts_data_func(binary_samples[binary_var_names], 
                                                    continuous_var_dim=continuous_var_dim, 
                                                    noise_std=noise_std)
    
    # Normalise data
    continuous_samples = continuous_samples / continuous_samples.std()

    binary_data_tensor = torch.tensor(binary_samples[['H', 'I', 'J']].values, dtype=torch.float32).to(device)
    continuous_data_tensor = torch.tensor(continuous_samples.values, dtype=torch.float32).to(device)
    input_dim = continuous_data_tensor.shape[1]
    output_dim = binary_data_tensor.shape[1]

    lr = 1e-2
    clip_grads = False
    randomly_permute_binaries = False
    epochs = 200
    beta = 0.0
    anneal_beta=True

    model = SimpleNN(input_dim, output_dim).to(device)
    train_vae(model, continuous_data_tensor, binary_data_tensor, learning_rate=lr, clip_grads=clip_grads,
              epochs=epochs, beta=beta, anneal_beta=anneal_beta)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass to get reconstructed output
    with torch.no_grad():
        yhat = model(continuous_data_tensor)
        yhat_df = pd.DataFrame(yhat.cpu().numpy())

        corrs_bin = pd.concat([binary_samples, yhat_df], axis=1).corr().loc[binary_samples.columns, yhat_df.columns]
        import pdb; pdb.set_trace()