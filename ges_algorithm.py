import networkx as nx
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import to_networkx_graph
from plotting_utils import plot_and_save_graph

def run_ges_algorithm(data, labels):
    """
    Runs the GES algorithm with the best parameters and returns the estimated causal graph.
    """
    # Best parameters found from tuning
    score_func = 'local_score_BDeu'
    maxP = None

    # Run GES with the best parameters
    ges_output = ges(data, score_func=score_func, maxP=maxP, parameters={})

    # Extract the estimated graph
    estimated_graph = ges_output['G']

    # Convert the CausalLearn GeneralGraph to NetworkX graph
    nx_graph = nx.DiGraph()
    for edge in estimated_graph.get_graph_edges():
        nx_graph.add_edge(edge.node1, edge.node2)

    # Relabel nodes using the provided labels
    nx_graph = nx.relabel_nodes(nx_graph, {i: labels[i] for i in range(len(labels))})

    # Plot and save the graph
    plot_and_save_graph(nx_graph, labels, 'ges_graph.png')
    
    return nx_graph

if __name__ == "__main__":
    from data_preparation import load_and_prepare_data
    file_path = 'data/student-por_raw.csv'
    df_encoded, labels, data = load_and_prepare_data(file_path)
    G_ges = run_ges_algorithm(data, labels)
    print("GES Algorithm graph created.")