import numpy as np
import pandas as pd
import torch
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_and_sample_bayesian_network(structure, cpds, sample_size=1000):
    """
    Create and sample from a Bayesian Network.

    Parameters:
    - structure: List of tuples representing edges in the Bayesian Network.
    - cpds: Dictionary where keys are variable names and values are TabularCPD objects.
    - sample_size: Number of samples to generate (default is 1000).

    Returns:
    - samples: DataFrame containing the sampled data.
    """
    # Create the Bayesian Network
    model = BayesianNetwork(structure)
    
    # Add nodes from CPDs if they are not already in the model
    for cpd in cpds.values():
        for var in cpd.variables:
            if not model.has_node(var):
                model.add_node(var)
    
    # Add CPDs to the model
    for cpd in cpds.values():
        model.add_cpds(cpd)

    # Check model validity
    if not model.check_model():
        raise ValueError("The model is invalid. Please check the structure and CPDs.")

    # Sample from the model
    sampler = BayesianModelSampling(model)
    samples = sampler.forward_sample(size=sample_size)
    return samples


def generate_continuous_variables(binary_data, continuous_var_dim, noise_std=0.1):
    """
    Generate continuous variables based on binary data using a neural network.

    Parameters:
    - binary_data: DataFrame containing binary samples.
    - continuous_var_dim: Dimensionality of the continuous variable.
    - noise_std: Standard deviation of the Gaussian noise added to the continuous variables (default is 0.1).

    Returns:
    - continuous_data: DataFrame containing the continuous variables.
    """
    binary_vars = binary_data.values
    continuous_data = np.zeros((binary_vars.shape[0], continuous_var_dim))

    # Define the neural network model
    model = MLP(binary_vars.shape[1], continuous_var_dim)
    
    # Use the binary variables as input to the neural network to generate continuous variables
    with torch.no_grad():
        inputs = torch.tensor(binary_vars, dtype=torch.float32)
        outputs = model(inputs).numpy()
    
    # Add Gaussian noise to the generated continuous variables
    continuous_data = outputs + np.random.normal(0, noise_std, outputs.shape)
    
    continuous_df = pd.DataFrame(continuous_data, columns=[f'Cont_Var_{i+1}' for i in range(continuous_var_dim)])
    return continuous_df


def generate_continuous_variables_simple(binary_data, continuous_var_dim, noise_std=0.1):
    """
    Generate continuous variables based on linear combinations and interaction terms.

    Parameters:
    - binary_data: DataFrame containing binary samples.
    - continuous_var_dim: Dimensionality of the continuous variable.
    - noise_std: Standard deviation of the Gaussian noise added to the continuous variables (default is 0.1).

    Returns:
    - continuous_data: DataFrame containing the continuous variables.
    """
    binary_vars = binary_data.values
    num_samples, num_binary_vars = binary_vars.shape
    
    # Generate random weights for linear combinations and interaction terms
    weights_linear = np.random.randn(num_binary_vars, continuous_var_dim)
    weights_interaction = np.random.randn(num_binary_vars, num_binary_vars, continuous_var_dim)
    
    continuous_data = np.zeros((num_samples, continuous_var_dim))
    
    # Compute linear combinations
    for i in range(continuous_var_dim):
        continuous_data[:, i] += np.dot(binary_vars, weights_linear[:, i])
        
        # Compute interaction terms
        for j in range(num_binary_vars):
            for k in range(j+1, num_binary_vars):
                continuous_data[:, i] += binary_vars[:, j] * binary_vars[:, k] * weights_interaction[j, k, i]
    
    # Add Gaussian noise to the generated continuous variables
    continuous_data += np.random.normal(0, noise_std, continuous_data.shape)
    
    continuous_df = pd.DataFrame(continuous_data, columns=[f'Cont_Var_{i+1}' for i in range(continuous_var_dim)])
    return continuous_df

def generate_continuous_variables_super_simple(binary_data, continuous_var_dim, noise_std=0.1):
    """
    Generate continuous variables based on linear combinations.

    Parameters:
    - binary_data: DataFrame containing binary samples.
    - continuous_var_dim: Dimensionality of the continuous variable.
    - noise_std: Standard deviation of the Gaussian noise added to the continuous variables (default is 0.1).

    Returns:
    - continuous_data: DataFrame containing the continuous variables.
    """
    binary_vars = binary_data.values
    num_samples, num_binary_vars = binary_vars.shape
    
    # Generate random weights for linear combinations and interaction terms
    weights_linear = np.random.randn(num_binary_vars, continuous_var_dim)
    
    continuous_data = np.zeros((num_samples, continuous_var_dim))
    
    # Compute linear combinations
    for i in range(continuous_var_dim):
        continuous_data[:, i] += np.dot(binary_vars, weights_linear[:, i])
        
    # Add Gaussian noise to the generated continuous variables
    continuous_data += np.random.normal(0, noise_std, continuous_data.shape)
    
    continuous_df = pd.DataFrame(continuous_data, columns=[f'Cont_Var_{i+1}' for i in range(continuous_var_dim)])
    return continuous_df