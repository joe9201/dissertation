from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam.utils import make_dot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx

def run_lingam_algorithm(data, labels):
    model_lingam = lingam.ICALiNGAM()
    model_lingam.fit(data)
    digraph = make_dot(model_lingam.adjacency_matrix_, labels=labels)
    digraph.render('lingam_graph', format='png')

    # Display the graph
    plt.imshow(mpimg.imread('lingam_graph.png'))
    plt.axis('off')
    plt.show()

    # Create NetworkX graph for evaluation
    adjacency_matrix = model_lingam.adjacency_matrix_
    lingam_graph = nx.DiGraph(adjacency_matrix)

    # Relabel nodes using the provided labels
    lingam_graph = nx.relabel_nodes(lingam_graph, {i: labels[i] for i in range(len(labels))})

    return lingam_graph

if __name__ == "__main__":
    from data_preparation import load_and_prepare_data
    file_path = 'data/student-por_raw.csv'
    df_encoded, labels, data = load_and_prepare_data(file_path)
    lingam_graph = run_lingam_algorithm(data, labels)
    print("LiNGAM Algorithm graph created.")