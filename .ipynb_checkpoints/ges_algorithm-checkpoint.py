import networkx as nx
import numpy as np
from causallearn.search.ScoreBased.GES import ges
from plotting_utils import plot_and_save_graph

def run_ges_algorithm(data, labels):
    """
    Runs the GES algorithm with the best parameters and returns the estimated causal graph.
    """
    score_func = 'local_score_BDeu'
    maxP = None

    # Convert data to matrix format as required by GES
    data_matrix = np.asmatrix(data)
    print("Data matrix shape:", data_matrix.shape)

    ges_output = ges(data_matrix, score_func=score_func, maxP=maxP, parameters={})

    # Extract the estimated graph
    estimated_graph = ges_output['G']

    # Convert the CausalLearn GeneralGraph to NetworkX graph
    nx_graph = nx.DiGraph()
    for edge in estimated_graph.get_graph_edges():
        nx_graph.add_edge(edge.node1, edge.node2)

    # Relabel nodes using the provided labels
    nx_graph = nx.relabel_nodes(nx_graph, {i: labels[i] for i in range(len(labels))})

    plot_and_save_graph(nx_graph, labels, 'ges_graph.png')
    
    return nx_graph