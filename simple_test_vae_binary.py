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

class BaseVAE(nn.Module, ABC):
    def __init__(self, input_dim, latent_dim):
        super(BaseVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, z):
        pass

    def sample_latent(self, probs):
        # Use the straight-through estimator for sampling
        # Sample from Bernoulli
        z = torch.bernoulli(probs)
        # Apply straight-through estimator
        z_straight_through = z + (probs - probs.detach())
        return z_straight_through

    def forward(self, x):
        probs = self.encode(x)
        z = self.sample_latent(probs)
        x_recon = self.decode(z)
        return x_recon, probs

    def sample(self, z):
        return self.decode(z)
    

class SuperSimpleBinaryVAE(BaseVAE):
    """
    Meant for use with generate_continuous_variables_super_simple datasets
    """
    def __init__(self, input_dim, latent_dim):
        super(SuperSimpleBinaryVAE, self).__init__(input_dim, latent_dim)

        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, input_dim)
        
    def encode(self, x):
        probs = F.sigmoid(self.fc1(x))
        return probs
    
    def decode(self, z):
        x_recon = self.fc2(z)
        return x_recon


class SimpleBinaryVAE(BaseVAE):
    """
    Meant for use with generate_continuous_variables_simple datasets
    """
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super(SimpleBinaryVAE, self).__init__(input_dim, latent_dim)
        self.hidden_dim = hidden_dim

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        x1 = F.relu(self.fc1(x))
        probs = F.sigmoid(self.fc2(x1))
        return probs
    
    def decode(self, z):
        x3 = F.relu(self.fc3(z))
        x_recon = self.fc4(x3)
        return x_recon


class MediumBinaryVAE(BaseVAE):
    """
    Meant for use with generate_continuous_variables_simple or generate_continuous_variables datasets
    """
    def __init__(self, input_dim, latent_dim, hidden_dim1=128, hidden_dim2=64):
        super(MediumBinaryVAE, self).__init__(input_dim, latent_dim)
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, latent_dim)

        # Decoder layers
        self.fc4 = nn.Linear(latent_dim, hidden_dim2)
        self.fc5 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc6 = nn.Linear(hidden_dim1, input_dim)
        
    def encode(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        probs = F.sigmoid(self.fc3(x2))
        return probs
    
    def decode(self, z):
        x4 = F.relu(self.fc4(z))
        x5 = F.relu(self.fc5(x4))
        x_recon = self.fc6(x5)
        return x_recon


def loss_function(x_recon, x, probs, beta=1.0):
    BCE = F.mse_loss(x_recon, x, reduction='sum')  # Reconstruction loss

    # janky KLD (penalises close to 0.5)
    KLD = (probs * (1 - probs)).sum()
    # print(f"BCE: {BCE}, KLD: {KLD}")
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
            x_recon, probs = model(x)
            loss = loss_function(x_recon, x, probs, beta)
            loss.backward()
            if clip_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.6f}')

def inspect_latent_space(model, data):
    model.eval()
    with torch.no_grad():
        probs = model.encode(data)
        z = (probs > 0.5).float()  # Binarize the latent variables
        return z.cpu().numpy()



# Define possible binary operations
operations = {
    'AND': lambda a, b: a & b,
    'OR': lambda a, b: a | b,
    'XOR': lambda a, b: a ^ b,
    'NAND': lambda a, b: ~(a & b) & 1,
    'NOR': lambda a, b: ~(a | b) & 1
}

# Function to check if a column can be expressed as a binary operation on df1
def check_operations(df1, df2_col):
    results = []
    for col1 in df1.columns:
        for col2 in df1.columns:
            if col1 == col2:
                continue
            for op_name, op_func in operations.items():
                if all(op_func(df1[col1], df1[col2]) == df2_col):
                    results.append((col1, col2, op_name))
    return results


if __name__ == "__main__":
    # Define the structure of the Bayesian Network
    structure = []

    # Define the CPDs (Conditional Probability Distributions)
    binary_var_names = ['H', 'I', 'J']
    # binary_var_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    cpds = {name: TabularCPD(variable=name, variable_card=2, values=[[0.5], [0.5]]) for name in binary_var_names}
 
    # Sample from the Bayesian Network
    print("Building binary samples...")
    binary_samples = create_and_sample_bayesian_network(structure, cpds, sample_size=10000)

    # Generate continuous variables
    print("\nBuilding continuous samples...")
    noise_std = 0
    continuous_var_dim = 20
    cts_data_func = generate_continuous_variables 
    # cts_data_func = generate_continuous_variables_simple
    # cts_data_func = generate_continuous_variables_super_simple
    continuous_samples = cts_data_func(binary_samples[binary_var_names], 
                                                       continuous_var_dim=continuous_var_dim, 
                                                       noise_std=noise_std)

    # Normalise data
    continuous_samples = continuous_samples / continuous_samples.std()

    continuous_data_tensor = torch.tensor(continuous_samples.values, dtype=torch.float32).to(device)
    input_dim = continuous_data_tensor.shape[1]
    # latent_dim = binary_samples.shape[1]
    latent_dim = len(binary_var_names)
    # model_type = SuperSimpleBinaryVAE
    # model_type = SimpleBinaryVAE
    model_type = MediumBinaryVAE
    model = model_type(input_dim, latent_dim).to(device)
    lr = 1e-3
    beta = 0
    clip_grads = True
    epochs = 100
    train_vae(model, continuous_data_tensor, epochs=epochs, 
              learning_rate=lr, beta=beta, clip_grads=clip_grads)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass to get reconstructed output and logits
    with torch.no_grad():
        x_recon, probs = model(continuous_data_tensor)
        x_recon_df = pd.DataFrame(x_recon)

        latent_space = inspect_latent_space(model, continuous_data_tensor)
        latent_space_df = pd.DataFrame(latent_space)

        corrs_binary = pd.concat([binary_samples, latent_space_df], axis=1).corr().loc[binary_samples.columns, latent_space_df.columns]
        props_binary = pd.DataFrame([[np.mean(binary_samples[col1] == latent_space_df[col2]) for col2 in latent_space_df.columns] for col1 in binary_samples.columns], index=binary_samples.columns, columns=latent_space_df.columns)
        corrs_cts = pd.concat([continuous_samples, x_recon_df], axis=1).corr().loc[continuous_samples.columns, x_recon_df.columns]

        unique_combos = pd.concat([binary_samples, latent_space_df], axis=1).astype('int').value_counts().reset_index().sort_values(['H', 'I', 'J'])
        info_recovered = len(unique_combos[latent_space_df.columns].value_counts()) == 2 ** len(binary_var_names)
        print(f"Binary proportions: \n {props_binary}")
        print(f"Information in binary variables {'recovered' if info_recovered else 'not recovered'}.")
        print(f"Binaries {'' if np.all([1.0 in props_binary[i].values for i in props_binary.columns]) else 'not'} perfectly recovered.")
        import pdb; pdb.set_trace()

"""
Next steps:
    - This is identifiable in the simplest case (or appears to be: latents always found)
        - Why is this? Is it because there is no invertible matrix that maps between binaries?
        - I think yes. Should be able to prove this.
    - For more complex relationships, it sometimes identifies, sometimes doesn't
        - I think this is because the information in binary A, B, C is the same as the information in
          some binary transformations of the binaries.
        - In the two variable case:
            - (A XOR B), B
            - (note, in 2 variable case we must always have one of the variables, or its negation)
        - Three variable case:
            - (A XOR B), B, C
            - (A XOR B), (B XOR C), C
            - (A XOR B), (B XOR C), NOT C etc

        - We sometimes observe more complicated operations:
            - ((A AND B) XOR C) OR (A AND B AND C)
            - A == B etc
        - How can we bias the model so it recovers OUR binary variables?? Is there a way to do this?
            - I think there may be no way to do this.
            - But what if we know there is a sparse causal structure between the variables??

"""