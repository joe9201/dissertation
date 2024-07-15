import sys
import networkx as nx
import matplotlib.pyplot as plt
from data_preparation import load_and_prepare_student_data, load_and_prepare_adult_data
from pc_algorithm import run_pc_algorithm
from fci_algorithm import run_fci_algorithm
from ges_algorithm import run_ges_algorithm
from lingam_algorithm import run_lingam_algorithm
from true_graph import create_true_graph_student, plot_true_graph
from causallearn.utils.GraphUtils import GraphUtils

# Check if the adult true graph function exists
try:
    from true_graph import create_true_graph_adult
except ImportError:
    create_true_graph_adult = None

def run_algorithms_for_dataset(data_preparation_func, true_graph_func, file_path):
    df_encoded, labels, data = data_preparation_func(file_path)
    
    # Run PC Algorithm
    PC_G = run_pc_algorithm(data, labels)
    
    # Run FCI Algorithm
    G_fci = run_fci_algorithm(data, labels)
    
    # Run GES Algorithm
    G_ges = run_ges_algorithm(data, labels)
    
    # Run LiNGAM Algorithm
    graph_dot_string = run_lingam_algorithm(data, labels)
    
    # Create and display the true graph
    if true_graph_func:
        G_true = true_graph_func()
        plot_true_graph(G_true, 'true_graph.png')
    else:
        G_true = None
        print("True graph function not available for this dataset.")
    
    return PC_G, G_fci, G_ges, graph_dot_string, G_true

if __name__ == "__main__":
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else 'student'

    datasets = {
        'student': {
            "file_path": 'data/student-por_raw.csv',
            "data_preparation_func": load_and_prepare_student_data,
            "true_graph_func": create_true_graph_student
        },
        'adult': {
            "file_path": 'data/adult.csv',
            "data_preparation_func": load_and_prepare_adult_data,
            "true_graph_func": create_true_graph_adult
        }
    }
    
    dataset = datasets.get(dataset_arg)
    
    if not dataset:
        print(f"Dataset '{dataset_arg}' not found. Available datasets are: {', '.join(datasets.keys())}")
        sys.exit(1)
    
    print(f"Running algorithms for dataset: {dataset['file_path']}")
    PC_G, G_fci, G_ges, graph_dot_string, G_true = run_algorithms_for_dataset(
        dataset['data_preparation_func'], 
        dataset['true_graph_func'], 
        dataset['file_path']
    )
    print(f"Algorithms completed for dataset: {dataset['file_path']}")