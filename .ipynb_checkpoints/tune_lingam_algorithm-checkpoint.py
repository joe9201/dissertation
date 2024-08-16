import numpy as np
import networkx as nx
from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam.utils import make_dot
from data_preparation import load_and_prepare_student_data
from true_graph import create_true_graph_student
from evaluation import evaluate_graph
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import traceback
import time
import os

# Set a consistent random seed for reproducibility
np.random.seed(42)

# Function to run the LiNGAM algorithm with specific parameters
def run_lingam_with_params(data, labels, max_iter):
    try:
        print(f"Running LiNGAM with max_iter={max_iter}")
        model_lingam = lingam.ICALiNGAM(max_iter=max_iter)
        model_lingam.fit(data)
        digraph = make_dot(model_lingam.adjacency_matrix_, labels=labels)
        
        # Ensure the correct file name and format
        file_name = f'lingam_graph_max_iter_{max_iter}'
        digraph.format = 'png'
        output_path = digraph.render(file_name, format='png')

        # Ensure the file is saved with the correct path
        png_path = f'{file_name}.png'

        # Wait until the file is completely written
        while not os.path.exists(png_path):
            time.sleep(0.1)

        # Read and display the graph image
        plt.imshow(mpimg.imread(png_path))
        plt.axis('off')
        plt.show()

        # Create NetworkX graph for evaluation
        adjacency_matrix = model_lingam.adjacency_matrix_
        lingam_graph = nx.DiGraph(adjacency_matrix)

        # Relabel nodes using the provided labels
        lingam_graph = nx.relabel_nodes(lingam_graph, {i: labels[i] for i in range(len(labels))})

        return lingam_graph
    except Exception as e:
        print(f"Error in run_lingam_with_params: max_iter={max_iter} - {str(e)}")
        traceback.print_exc()  # Print the full traceback for detailed debugging
        raise

# Function to perform grid search for LiNGAM algorithm
def grid_search_lingam(data, labels, true_graph):
    param_grid = {
        'max_iter': [500, 1000, 1500, 2000, 2500]
    }

    best_params = None
    best_score = float('inf')

    for max_iter in param_grid['max_iter']:
        try:
            nx_graph = run_lingam_with_params(data, labels, max_iter)
            shd, recall, precision = evaluate_graph(nx_graph, true_graph)
            score = shd  # Using SHD as the score metric

            if score < best_score:
                best_score = score
                best_params = {'max_iter': max_iter}

            print(f"Params: max_iter={max_iter} - SHD: {shd}, Recall: {recall}, Precision: {precision}")
        except Exception as e:
            print(f"Error with params: max_iter={max_iter} - {str(e)}")
            traceback.print_exc()  # Print the full traceback for detailed debugging

    print(f"Best parameters for LiNGAM Algorithm: {best_params}")
    print(f"Best score: {best_score}")

# Main function to load data and perform grid search
if __name__ == "__main__":
    file_path = 'data/student-por_raw.csv'
    df_encoded, labels, data = load_and_prepare_student_data(file_path)
    true_graph = create_true_graph_student()
    grid_search_lingam(data, labels, true_graph)