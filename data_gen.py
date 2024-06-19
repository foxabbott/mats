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

    # Sample from the Bayesian Network
    print("Building binary samples...")
    binary_samples = create_and_sample_bayesian_network(structure, cpds, sample_size=10000)

    # Generate continuous variables
    print("\nBuilding continuous samples...")
    continuous_samples = generate_continuous_variables(binary_samples[['H', 'I', 'J']], continuous_var_dim=20, noise_std=0.1)
    print(continuous_samples.head())
    import pdb; pdb.set_trace()
