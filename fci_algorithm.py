import networkx as nx
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from plotting_utils import plot_and_save_graph

def run_fci_algorithm(data, labels):
    """Runs the FCI algorithm and returns the estimated causal graph."""
    cg_fci = fci(data, alpha=0.01, indep_test=fisherz, stable=True, uc_rule=0)

    # Convert CausalLearn Graph to NetworkX graph (updated method)
    nx_graph = nx.DiGraph(cg_fci[0].graph)

    # Relabel nodes using the provided labels (if needed)
    nx_graph = nx.relabel_nodes(nx_graph, {i: labels[i] for i in range(len(labels))})

    plot_and_save_graph(nx_graph, labels, 'fci_graph.png')
    
    return nx_graph