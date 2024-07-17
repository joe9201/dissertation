import networkx as nx
from causallearn.search.ScoreBased.GES import ges
from plotting_utils import plot_and_save_graph, causal_learn_to_networkx

def run_ges_algorithm(data, labels):
    """Runs the GES algorithm and returns the estimated causal graph."""
    record_ges = ges(data)

    # Convert CausalLearn Graph to NetworkX graph
    ges_graph = record_ges['G']
    nx_graph = causal_learn_to_networkx(ges_graph)

    # If the labels are not already strings, convert them
    if not all(isinstance(label, str) for label in labels):
        labels = [str(label) for label in labels]

    # Create a dictionary to map nodes to their labels
    node_label_map = {i: label for i, label in enumerate(labels)}

    # Relabel nodes using the provided labels
    nx_graph = nx.relabel_nodes(nx_graph, node_label_map)

    # Plot and save the graph
    plot_and_save_graph(nx_graph, labels, 'ges_graph.png')

    return nx_graph

if __name__ == "__main__":
    from data_preparation import load_and_prepare_data
    file_path = 'data/student-por_raw.csv'
    df_encoded, labels, data = load_and_prepare_data(file_path)
    G_ges = run_ges_algorithm(data, labels)
    print("GES Algorithm graph created.")