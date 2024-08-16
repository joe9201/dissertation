import networkx as nx

def calculate_shd(true_graph, estimated_graph):
    true_edges = set(true_graph.edges())
    estimated_edges = set(estimated_graph.edges())
    
    extra_edges = estimated_edges - true_edges
    missing_edges = true_edges - estimated_edges
    
    shd = len(extra_edges) + len(missing_edges)
    return shd

def evaluate_graph(estimated_graph, true_graph):
    shd = calculate_shd(true_graph, estimated_graph)
    return shd