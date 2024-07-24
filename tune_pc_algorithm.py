import numpy as np
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from data_preparation import load_and_prepare_student_data
from true_graph import create_true_graph_student
from evaluation import evaluate_graph
from plotting_utils import plot_and_save_graph
import traceback

# Function to run the PC algorithm with specific parameters
def run_pc_with_params(data, labels, alpha, stable, uc_rule):
    try:
        print(f"Running PC with alpha={alpha}, stable={stable}, uc_rule={uc_rule}")
        cg_pc = pc(data, alpha=alpha, indep_test=fisherz, stable=stable, uc_rule=uc_rule)

        # Convert CausalLearn Graph to NetworkX graph
        nx_graph = nx.DiGraph(cg_pc.G.graph)

        # Relabel nodes using the provided labels
        nx_graph = nx.relabel_nodes(nx_graph, {i: labels[i] for i in range(len(labels))})

        # Plot and save the graph
        plot_and_save_graph(nx_graph, labels, f'pc_graph_alpha_{alpha}_stable_{stable}_uc_rule_{uc_rule}.png')

        return nx_graph
    except Exception as e:
        print(f"Error in run_pc_with_params: alpha={alpha}, stable={stable}, uc_rule={uc_rule} - {str(e)}")
        traceback.print_exc()  # Print the full traceback for detailed debugging
        raise

# Function to perform grid search for PC algorithm
def grid_search_pc(data, labels, true_graph):
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
                try:
                    nx_graph = run_pc_with_params(data, labels, alpha, stable, uc_rule)
                    shd, recall, precision = evaluate_graph(nx_graph, true_graph)
                    score = shd  # Using SHD as the score metric

                    if score < best_score:
                        best_score = score
                        best_params = {'alpha': alpha, 'stable': stable, 'uc_rule': uc_rule}

                    print(f"Params: alpha={alpha}, stable={stable}, uc_rule={uc_rule} - SHD: {shd}, Recall: {recall}, Precision: {precision}")
                except Exception as e:
                    print(f"Error with params: alpha={alpha}, stable={stable}, uc_rule={uc_rule} - {str(e)}")
                    traceback.print_exc()  # Print the full traceback for detailed debugging

    print(f"Best parameters for PC Algorithm: {best_params}")
    print(f"Best score: {best_score}")

# Main function to load data and perform grid search
if __name__ == "__main__":
    file_path = 'data/student-por_raw.csv'
    df_encoded, labels, data = load_and_prepare_student_data(file_path)
    true_graph = create_true_graph_student()
    grid_search_pc(data, labels, true_graph)