import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD
from data_gen import create_and_sample_bayesian_network


def learn_bayesian_network(data):
    """
    Learn the structure and CPDs of a Bayesian Network from a dataset.

    Parameters:
    - data: DataFrame containing the dataset.

    Returns:
    - model: The learned Bayesian Network model.
    - cpds: Dictionary of CPDs learned from the data.
    """
    # Learn the structure of the Bayesian Network
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=BicScore(data))

    # Create the Bayesian Network with the learned structure
    model = BayesianNetwork(best_model.edges())

    # Learn the CPDs for the model
    model.fit(data, estimator=MaximumLikelihoodEstimator)

    # Extract the CPDs from the model
    cpds = {cpd.variable: cpd for cpd in model.get_cpds()}

    return model, cpds

if __name__ == "__main__":
    # Example usage with generated data
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

    # Generate samples using the previous function
    print("Generating samples...")
    samples = create_and_sample_bayesian_network(structure, cpds, sample_size=10000)

    # Learn the Bayesian Network from the samples
    print("Learning model...")
    learned_model, learned_cpds = learn_bayesian_network(samples)

    # Print the learned structure
    print("Learned Structure:")
    print(learned_model.edges())

    # Print the learned CPDs
    print("\nLearned CPDs:")
    for var, cpd in learned_cpds.items():
        print(f"\nCPD of {var}:")
        print(cpd)
