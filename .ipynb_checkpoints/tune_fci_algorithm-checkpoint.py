import numpy as np
import networkx as nx
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from data_preparation import load_and_prepare_student_data
from true_graph import create_true_graph_student
from evaluation import evaluate_graph

# Function to run the FCI algorithm with specific parameters
def run_fci_with_params(data, labels, alpha, stable, uc_rule):
    cg_fci = fci(data, alpha=alpha, indep_test=fisherz, stable=stable, uc_rule=uc_rule)
    nx_graph = nx.DiGraph(cg_fci[0].graph)
    nx_graph = nx.relabel_nodes(nx_graph, {i: labels[i] for i in range(len(labels))})
    return nx_graph

# Function to perform grid search for FCI algorithm
def grid_search_fci(data, labels, true_graph):
    param_grid = {
        'alpha': [0.01, 0.05, 0.1],
        'stable': [True, False],
        'uc_rule': [0, 1, 2]
    }

    best_params = None
    best_score = float('inf')

    for alpha in param_grid['alpha']:
        for stable in param_grid['stable']:
            for uc_rule in param_grid['uc_rule']:
                nx_graph = run_fci_with_params(data, labels, alpha, stable, uc_rule)
                shd, recall, precision = evaluate_graph(nx_graph, true_graph)
                score = shd  # You can use a combination of metrics to define the score

                if score < best_score:
                    best_score = score
                    best_params = {'alpha': alpha, 'stable': stable, 'uc_rule': uc_rule}

                print(f"Params: alpha={alpha}, stable={stable}, uc_rule={uc_rule} - SHD: {shd}, Recall: {recall}, Precision: {precision}")

    print(f"Best parameters for FCI Algorithm: {best_params}")
    print(f"Best score: {best_score}")

# Main function to load data and perform grid search
if __name__ == "__main__":
    file_path = 'data/student-por_raw.csv'
    df_encoded, labels, data = load_and_prepare_student_data(file_path)
    true_graph = create_true_graph_student()
    grid_search_fci(data, labels, true_graph)