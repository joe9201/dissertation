import networkx as nx
from causallearn.search.ScoreBased.GES import ges
from plotting_utils import plot_and_save_graph

def run_ges_algorithm(data, labels):
    record_ges = ges(data)
    plot_and_save_graph(record_ges['G'], labels, 'ges_graph.png')
    
    ges_graph = record_ges['G']
    G_ges = nx.DiGraph()
    for node in ges_graph.nodes:
        G_ges.add_node(node.get_name())

    for i in range(len(ges_graph.graph)):
        for j in range(len(ges_graph.graph)):
            if ges_graph.graph[i, j] > 0:  # Check for a directed edge
                G_ges.add_edge(ges_graph.nodes[i].get_name(), ges_graph.nodes[j].get_name())

    return G_ges

if __name__ == "__main__":
    from data_preparation import load_and_prepare_data
    file_path = 'data/student-por_raw.csv'
    df_encoded, labels, data = load_and_prepare_data(file_path)
    G_ges = run_ges_algorithm(data, labels)
    print("GES Algorithm graph created.")