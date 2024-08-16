import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from networkx.drawing.nx_pydot import to_pydot
import networkx as nx
from causallearn.graph.GeneralGraph import GeneralGraph

def plot_and_save_graph(graph, labels, filename):
    graph_copy = nx.DiGraph(graph)

    # Set labels as attributes
    for i, node in enumerate(graph_copy.nodes()):
        graph_copy.nodes[node]['label'] = labels[i]

    pyd = to_pydot(graph_copy)

    for node in pyd.get_nodes():
        node.set_fontsize(12)
    for edge in pyd.get_edges():
        edge.set_penwidth(2)

    # Plot and save
    plt.figure(figsize=(10, 8))
    pyd.write_png(filename, prog='dot')

    # Display the graph with pause
    img = mpimg.imread(filename)
    plt.axis('off')
    plt.imshow(img)
    plt.show(block=False)
    plt.close()

def causal_learn_to_networkx(causal_learn_graph):
    nx_graph = nx.DiGraph()

    for edge in causal_learn_graph.get_graph_edges():
        nx_graph.add_edge(edge.node1, edge.node2)

    return nx_graph
