import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
from causallearn.utils.GraphUtils import GraphUtils

def plot_and_save_graph(graph, labels, filename):
    pyd = GraphUtils.to_pydot(graph, labels=labels)
    img = mpimg.imread(io.BytesIO(pyd.create_png(f="png")), format='png')
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(filename, format='png')
    plt.show()