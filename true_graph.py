import networkx as nx
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt
import io
import matplotlib.image as mpimg
from networkx.drawing.nx_pydot import to_pydot

def create_true_graph_student():
    G_true = nx.DiGraph()

    # Node positions (using tuples for coordinates)
    pos = {
        "G_avg": (0.176, -0.450),
        "Medu": (-0.630, 0.589),
        "Pstatus": (-1.493, -0.841),
        "absences": (-1.468, 0.878),
        "address": (-1.971, 1.405),
        "failures": (1.150, 1.680),
        "famrel": (-2.149, -1.148),
        "famsup": (-1.049, -0.302),
        "health": (-0.688, 0.017),
        "higher": (0.155, 0.551),
        "internet": (0.647, 1.138),
        "paid": (-0.463, -1.386),
        "schoolsup": (1.035, 0.158),
        "studytime": (0.302, -1.504)
    }

    # Add nodes with positions
    for node, position in pos.items():
        G_true.add_node(node, pos=position)

    # Add edges
    G_true.add_edges_from([
        ("Medu", "G_avg"),
        ("Medu", "absences"),
        ("Medu", "higher"),
        ("Pstatus", "G_avg"),
        ("Pstatus", "absences"),
        ("Pstatus", "famrel"),
        ("address", "absences"),
        ("failures", "G_avg"),
        ("failures", "absences"),
        ("famsup", "G_avg"),
        ("famsup", "absences"),
        ("health", "G_avg"),
        ("health", "absences"),
        ("higher", "G_avg"),
        ("internet", "G_avg"),
        ("internet", "absences"),
        ("paid", "G_avg"),
        ("schoolsup", "G_avg"),
        ("studytime", "G_avg")
    ])

    return G_true

def plot_true_graph(G_true, filename):
    pyd = to_pydot(G_true)

    # Save the image in the vector format
    pyd.write_png(filename)

    # Read and display the image
    img = mpimg.imread(filename)
    plt.axis('off')
    plt.imshow(img)
    plt.show(block=False) 

if __name__ == "__main__":
    G_true = create_true_graph_student()
    plot_true_graph(G_true, 'true_graph.png')