import networkx as nx
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt
import io
from networkx.drawing.nx_pydot import to_pydot
import matplotlib.image as mpimg

def create_true_graph_student():
    G_true = nx.DiGraph()

    pos = {
        "G_avg": (0.207, -0.688),
        "Medu": (-0.641, 0.589),
        "Pstatus": (-1.699, -0.614),
        "absences": (-1.531, 0.937),
        "address": (-2.008, 1.450),
        "failures": (0.873, 1.702),
        "famrel": (-2.270, -0.747),
        "famsup": (-1.316, -0.265),
        "health": (-0.945, 0.076),
        "higher": (-0.206, 0.589), 
        "internet": (0.302, 1.034),
        "paid": (-0.243, -1.400),
        "schoolsup": (0.867, -0.176),
        "studytime": (0.799, -1.289)
    }

    for node, position in pos.items():
        G_true.add_node(node, pos=position)

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


def plot_true_graph(G_true, dataset_name):
    filename = f"{dataset_name}_true_graph.png"
    pyd = to_pydot(G_true)

    for node in pyd.get_nodes():
        node.set_fontsize(12) 

    for edge in pyd.get_edges():
        edge.set_penwidth(2)
        
    plt.figure(figsize=(10,8))

    pyd.write_png(filename) 

    img = mpimg.imread(filename)
    plt.axis('off')
    plt.imshow(img)
    plt.show(block=False)

if __name__ == "__main__":
    G_true = create_true_graph_student()
    plot_true_graph(G_true, "student")