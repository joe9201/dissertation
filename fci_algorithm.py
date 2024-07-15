import networkx as nx
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from plotting_utils import plot_and_save_graph

def run_fci_algorithm(data, labels):
    cg_fci = fci(data, alpha=0.1, labels=labels, indep_test=fisherz, stable=True, uc_rule=0)
    plot_and_save_graph(cg_fci[0], labels, 'fci_graph.png')

    fci_graph = cg_fci[0]
    G_fci = nx.DiGraph()
    for node in fci_graph.nodes:
        G_fci.add_node(node.get_name())

    for i in range(len(fci_graph.graph)):
        for j in range(len(fci_graph.graph)):
            if fci_graph.graph[i, j] > 0:  # Check for a directed edge
                G_fci.add_edge(fci_graph.nodes[i].get_name(), fci_graph.nodes[j].get_name())

    return G_fci

if __name__ == "__main__":
    from data_preparation import load_and_prepare_data
    file_path = 'data/student-por_raw.csv'
    df_encoded, labels, data = load_and_prepare_data(file_path)
    G_fci = run_fci_algorithm(data, labels)
    print("FCI Algorithm graph created.")