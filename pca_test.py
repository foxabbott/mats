import pandas as pd
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from data_gen import generate_continuous_variables, generate_continuous_variables_simple, generate_continuous_variables_super_simple, create_and_sample_bayesian_network
from sklearn.decomposition import PCA

def pca(continuous_samples, binary_samples, k=3):
    # Standardize the data
    x_centered = continuous_samples - np.mean(continuous_samples, axis=0)

    # Perform PCA
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(x_centered)

    # The transformed data (x_pca) corresponds to the recovered y up to a linear transformation
    recovered_binaries = x_pca
    y_hat = pd.DataFrame(recovered_binaries)
    corrs = pd.concat([y_hat, binary_samples], axis=1).corr().loc[y_hat.columns, binary_samples.columns]
    import pdb; pdb.set_trace()

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
    noise_std = 0.0
    continuous_var_dim = 10
    continuous_generation_function = generate_continuous_variables_super_simple
    continuous_samples = continuous_generation_function(binary_samples[['H', 'I', 'J']], 
                                                       continuous_var_dim=continuous_var_dim, 
                                                       noise_std=noise_std)

    # Normalise data
    continuous_samples = continuous_samples / continuous_samples.std()

    pca(continuous_samples, binary_samples)
    import pdb; pdb.set_trace()