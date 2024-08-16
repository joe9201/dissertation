import numpy as np
import pandas as pd
from lingam.direct_lingam import DirectLiNGAM
from lingam.utils import make_dot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time
import traceback
import networkx as nx

def run_direct_lingam_with_params(data: pd.DataFrame, labels: list, measure: str):
    """
    Run the DirectLiNGAM algorithm with specific parameters.

    Parameters:
    data (pd.DataFrame): The input data for causal discovery.
    labels (list): List of labels for the data columns.
    measure (str): Measure to evaluate independence ('pwling', 'kernel', 'pwling_fast').

    Returns:
    graph: The adjacency matrix representing the causal graph.
    dot: The DOT graph for visualization.
    """
    try:
        print(f"Running DirectLiNGAM with measure={measure}")
        model = DirectLiNGAM(measure=measure)
        model.fit(data)
        adjacency_matrix = model.adjacency_matrix_

        print(f"Adjacency Matrix:\n{adjacency_matrix}")

        # Create a causal diagram
        dot = make_dot(adjacency_matrix, labels=labels)
        
        # Ensure the correct file name and format
        file_name = f'direct_lingam_graph_measure_{measure}'
        dot.format = 'png'
        output_path = dot.render(file_name, format='png')

        # Ensure the file is saved with the correct path
        png_path = f'{file_name}.png'

        # Wait until the file is completely written
        while not os.path.exists(png_path):
            time.sleep(0.1)

        print(f"Graph saved at {png_path}")

        # Read and display the graph image
        img = mpimg.imread(png_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        # Create NetworkX graph for evaluation
        direct_lingam_graph = nx.DiGraph(adjacency_matrix)

        # Relabel nodes using the provided labels
        direct_lingam_graph = nx.relabel_nodes(direct_lingam_graph, {i: labels[i] for i in range(len(labels))})

        return direct_lingam_graph
    except Exception as e:
        print(f"Error in run_direct_lingam_with_params: measure={measure} - {str(e)}")
        traceback.print_exc()  # Print the full traceback for detailed debugging
        raise

if __name__ == "__main__":
    from data_preparation import load_and_prepare_student_data
    
    try:
        file_path = 'data/student-por_raw.csv'
        df_encoded, labels, data = load_and_prepare_student_data(file_path)
        print("Data loaded successfully")

        # Test DirectLiNGAM with 'pwling' measure
        graph = run_direct_lingam_with_params(data, labels, measure='pwling')
        print("DirectLiNGAM Algorithm graph created successfully.")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
