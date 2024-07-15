import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from plotting_utils import plot_and_save_graph

def run_pc_algorithm(data, labels):
    cg_pc = pc(data, alpha=0.1, labels=labels)
    plot_and_save_graph(cg_pc.G, labels, 'pc_graph.png')
    
    PC_G = nx.DiGraph()
    for node in cg_pc.G.nodes:
        PC_G.add_node(node.get_name())

    for i in range(len(cg_pc.G.graph)):
        for j in range(len(cg_pc.G.graph)):
            if cg_pc.G.graph[i, j] > 0:  # Check for a directed edge
                PC_G.add_edge(cg_pc.G.nodes[i].get_name(), cg_pc.G.nodes[j].get_name())

    return PC_G

if __name__ == "__main__":
    from data_preparation import load_and_prepare_data
    file_path = 'data/student-por_raw.csv'
    df_encoded, labels, data = load_and_prepare_data(file_path)
    PC_G = run_pc_algorithm(data, labels)
    print("PC Algorithm graph created.")