import numpy as np
import networkx as nx
from causallearn.search.ScoreBased.GES import ges
from data_preparation import load_and_prepare_student_data
from true_graph import create_true_graph_student
from evaluation import evaluate_graph
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plotting_utils import plot_and_save_graph
import traceback

# Function to run the GES algorithm with specific parameters
def run_ges_with_params(data, labels, score_func, maxP):
    try:
        print(f"Running GES with score_func={score_func} and maxP={maxP}")
        ges_output = ges(data, score_func=score_func, maxP=maxP, parameters={})

        if 'G' not in ges_output:
            raise ValueError("GES function did not return expected output.")

        estimated_graph = ges_output['G']

        # Convert the CausalLearn GeneralGraph to NetworkX graph
        nx_graph = nx.DiGraph()
        for edge in estimated_graph.get_graph_edges():
            nx_graph.add_edge(edge.node1, edge.node2)

        # Relabel nodes using the provided labels
        nx_graph = nx.relabel_nodes(nx_graph, {i: labels[i] for i in range(len(labels))})

        # Plot and save the graph
        plot_and_save_graph(nx_graph, labels, f'ges_graph_{score_func}_{maxP}.png')

        return nx_graph
    except Exception as e:
        print(f"Error in run_ges_with_params: score_func={score_func}, maxP={maxP} - {str(e)}")
        traceback.print_exc()  # Print the full traceback for detailed debugging
        raise

# Function to perform grid search for GES algorithm
def grid_search_ges(data, labels, true_graph):
    param_grid = {
        'score_func': ['local_score_BIC', 'local_score_BDeu'],
        'maxP': [None, 2, 3, 4]
    }

    best_params = None
    best_score = float('inf')

    for score_func in param_grid['score_func']:
        for maxP in param_grid['maxP']:
            try:
                nx_graph = run_ges_with_params(data, labels, score_func, maxP)
                shd, recall, precision = evaluate_graph(nx_graph, true_graph)
                score = shd  # Using SHD as the score metric

                if score < best_score:
                    best_score = score
                    best_params = {'score_func': score_func, 'maxP': maxP}

                print(f"Params: score_func={score_func}, maxP={maxP} - SHD: {shd}, Recall: {recall}, Precision: {precision}")
            except Exception as e:
                print(f"Error with params: score_func={score_func}, maxP={maxP} - {str(e)}")
                traceback.print_exc()  # Print the full traceback for detailed debugging

    print(f"Best parameters for GES Algorithm: {best_params}")
    print(f"Best score: {best_score}")

# Main function to load data and perform grid search
if __name__ == "__main__":
    file_path = 'data/student-por_raw.csv'
    df_encoded, labels, data = load_and_prepare_student_data(file_path)
    true_graph = create_true_graph_student()
    grid_search_ges(data, labels, true_graph)
