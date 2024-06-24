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


def recover_A(binary, continuous):
    A = np.linalg.inv(binary.T @ binary) @ binary.T @ continuous
    import pdb; pdb.set_trace()
    return A


if __name__ == "__main__":
    # Define the structure of the Bayesian Network
    structure = []

    # Define the CPDs (Conditional Probability Distributions)
    # binary_var_names = ['H', 'I', 'J']
    binary_var_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    cpds = {name: TabularCPD(variable=name, variable_card=2, values=[[0.5], [0.5]]) for name in binary_var_names}
 
    # Sample from the Bayesian Network
    print("Building binary samples...")
    sample_size = 1000
    binary_samples = create_and_sample_bayesian_network(structure, cpds, sample_size=sample_size)

    # Generate continuous variables
    print("\nBuilding continuous samples...")
    noise_std = 0
    continuous_var_dim = 3
    # cts_data_func = generate_continuous_variables 
    # cts_data_func = generate_continuous_variables_simple
    cts_data_func = generate_continuous_variables_super_simple
    continuous_samples = cts_data_func(binary_samples[binary_var_names], 
                                                       continuous_var_dim=continuous_var_dim, 
                                                       noise_std=noise_std)
    
    # Normalise data
    continuous_samples = continuous_samples / continuous_samples.std()
    recover_A(binary_samples, continuous_samples)