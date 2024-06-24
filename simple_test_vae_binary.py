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


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = torch.zeros_like(y).scatter_(-1, y.max(-1, keepdim=True)[1], 1.0)
        y = (y_hard - y).detach() + y
    return y

def probabilities_to_logits(probs):
    # Add a small epsilon to avoid log(0) or log(1)
    eps = 1e-7
    probs = torch.clamp(probs, eps, 1 - eps)
    return torch.log(probs / (1 - probs))


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

    def forward(self, x, y):
        probs = self.encode(x)
        # Instead, insert true latents inserted (test):
        # if random.random() < 0.01:
            # probs = y
        z = self.sample_latent(probs)
        # logits = probabilities_to_logits(probs)
        # z = gumbel_softmax(logits, temperature=1.0, hard=True)
        x_recon = self.decode(z)
        return x_recon, probs

    def sample(self, z):
        return self.decode(z)
    

class SuperSimpleBinaryVAE(BaseVAE):
    """
    Meant for use with generate_continuous_variables_super_simple datasets
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_prob=0.1):
        super(SuperSimpleBinaryVAE, self).__init__(input_dim, latent_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, input_dim)
        
    def encode(self, x):
        z1 = F.relu(self.fc1(x))
        # z2 = self.dropout1(z1)
        probs = torch.sigmoid(self.fc2(z1))
        # plt.scatter(x=pd.DataFrame(x), y=pd.DataFrame(probs.detach().numpy())[3])
        # import pdb; pdb.set_trace()
        return probs
    
    def decode(self, z):
        # z = self.dropout2(z)
        x_recon = self.fc3(z)
        return x_recon

    # def forward(self, x):
    #     z = self.encode(x)
    #     x_recon = self.decode(z)
    #     return x_recon, z


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
        probs = torch.sigmoid(self.fc2(x1))
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
        probs = torch.sigmoid(self.fc3(x2))
        return probs
    
    def decode(self, z):
        x4 = F.relu(self.fc4(z))
        x5 = F.relu(self.fc5(x4))
        x_recon = self.fc6(x5)
        return x_recon


def loss_function(x_recon, x, probs, beta=1.0):
    BCE = F.mse_loss(x_recon, x, reduction='sum') / x.shape[1] # Reconstruction loss
    # dividing by x.shape[1] so that relative magnitudes of the two terms remain similar for different latent space dimensions

    # janky KLD (penalises close to 0.5)
    KLD = (probs * (1 - probs)).sum() / x.shape[1]

    # Entropy Regularization
    # epsilon = 1e-10  # Small value to prevent log(0)
    # entropy = - (probs * torch.log(probs + epsilon) + (1 - probs) * torch.log(1 - probs + epsilon)).mean()

    # probs_mean = probs.mean(dim=0)
    # variance_penalty = (probs_mean - 0.5).pow(2).sum()

    # print(f"BCE: {BCE}, KLD: {KLD}, Var: {variance_penalty}")
    
    return BCE + beta * KLD
    # return BCE + beta * entropy
    # return BCE + beta * KLD + 100 * variance_penalty

def train_vae(model, cts_data, bin_data, epochs=100, batch_size=256, learning_rate=1e-3, beta=1.0, 
              clip_grads=False, anneal_beta=False):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=lr*10, step_size_up=1000, mode='triangular2', cycle_momentum=False)

    train_loader = DataLoader(TensorDataset(cts_data, bin_data), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        if anneal_beta:
            beta_epoch = beta * epoch / epochs
        else:
            beta_epoch = beta

        train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, probs = model(x, y)
            loss = loss_function(x_recon, x, probs, beta_epoch)
            loss.backward()
            if clip_grads:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_loss += loss.item()
            optimizer.step()
            # scheduler.step()

        if cts_data.shape[1] == 1:
            plot_encoded_vars(cts_data, epoch)

        print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.6f}')


def inspect_latent_space(model, data):
    model.eval()
    with torch.no_grad():
        probs = model.encode(data)
        z = (probs > 0.5).float()  # Binarize the latent variables
        return z.cpu().numpy()

def plot_encoded_vars(data, epoch):
    input_vals = np.arange(data.min(), data.max(), 0.01).astype(np.float32)
    encodings = pd.DataFrame(model.encode(torch.tensor(input_vals).view(-1, 1)).detach().numpy())
    encodings['input'] = input_vals
    plt.figure(figsize=(10, 6))
    for column in encodings.columns[:-1]:  # Exclude the 'input' column
        plt.plot(encodings['input'], encodings[column], label=f'Encoding {column}')
    for value in [pd.DataFrame(data).value_counts().index[i][0] for i in range(len(pd.DataFrame(data).value_counts().index))] :
        plt.axvline(x=value, color='r', linestyle='--', linewidth=1)
    plt.savefig(f"plots/simple_run/{epoch}.jpg")
    plt.close()


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0, std=0.1)  # Increase std for more chaos
        torch.nn.init.zeros_(m.bias)


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
    # Note: learning seems really hard when this is th

    continuous_data_tensor = torch.tensor(continuous_samples.values, dtype=torch.float32).to(device)
    input_dim = continuous_data_tensor.shape[1]
    # latent_dim = binary_samples.shape[1]
    latent_dim = len(binary_var_names)
    model_type = SuperSimpleBinaryVAE
    # model_type = SimpleBinaryVAE
    # model_type = MediumBinaryVAE
    lr = 1e-3
    beta = 1.0
    anneal_beta = True
    clip_grads = False
    epochs = 200
    hidden_dim = 2**9

    model = model_type(input_dim, hidden_dim, latent_dim).to(device)
    # model.apply(init_weights)
    
    if continuous_var_dim == 1:
        plot_encoded_vars(continuous_data_tensor, "__initialised")

    train_vae(model, continuous_data_tensor, binary_data_tensor, epochs=epochs, 
            learning_rate=lr, beta=beta, clip_grads=clip_grads,
            anneal_beta=anneal_beta)

    # Set the model to evaluation mode
    model.eval()

    # Forward pass to get reconstructed output and logits
    with torch.no_grad():
        x_recon, probs = model(continuous_data_tensor, binary_data_tensor)
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
        print(f"beta: {beta}")
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
    - Also: this doesn't seem to work if the number of observed variables is lower than the 
      number of binaries. This isn't surprising. But is problematic if we want to i.e. use a 5-token 
      string to infer 1000 binaries... Even if we train it on really long strings...?


    - beta value seems very important for good training. Too high, and it gets stuck in a bad set of binaries.
      Keep around 0.1???

    - 22/06/24 Current thoughts: Maybe non-identifiability as described above is not a problem because we 
        expect the concepts to be linearly represented anyway. So we can stick to a simple function.
        Even in the more complex case, perhaps adding a sparsity penalty could help?? Will more entangled 
        binaries be activated more often? Think about this, if you decide on a more complex architecture.

        Important: when the number of inputs is small, the encoder is much too simple to be able to map
        properly. Think more about why this is, and if there's a way to fix it.
"""