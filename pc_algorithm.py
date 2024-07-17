import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from plotting_utils import plot_and_save_graph

def run_pc_algorithm(data, labels):
    """
    Runs the PC algorithm and returns the estimated causal graph.
    """
    cg_pc = pc(data, alpha=0.05, indep_test=fisherz, stable=True, uc_rule=0)

    # Convert CausalLearn Graph to NetworkX graph (updated method)
    nx_graph = nx.DiGraph(cg_pc.G.graph)

    # Relabel nodes using the provided labels (if needed)
    nx_graph = nx.relabel_nodes(nx_graph, {i: labels[i] for i in range(len(labels))})

    plot_and_save_graph(nx_graph, labels, 'pc_graph.png')
    
    return nx_graph