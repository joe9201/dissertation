import networkx as nx
import matplotlib.pyplot as plt

def create_true_graph_student():
    G_true = nx.DiGraph()

    pos = {
        "G_avg": (0.207, -0.688),
        "Medu": (-0.641, 0.589),
        "Pstatus": (-1.699, -0.614),
        "absences": (-1.531, 0.937),
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
        ("studytime", "G_avg"),
        ("absences", "G_avg")
    ])

    return G_true

def create_true_graph_student_small():
    G_true = nx.DiGraph()

    pos = {
        "absences": (0.643, 0.440),
        "Medu": (-0.641, 0.589),
        "higher": (-1.0, 0.273),
        "studytime": (0.378, 0.250),
        "health": (0.517, 0.179),
        "failures": (0.742, 0.182),
        "G_avg": (0.314, 0.427)
    }

    for node, position in pos.items():
        G_true.add_node(node, pos=position)

    G_true.add_edges_from([
        ("Medu", "higher"),
        ("higher", "G_avg"),
        ("studytime", "G_avg"),
        ("health", "G_avg"),
        ("health", "absences"),
        ("failures", "G_avg"),
        ("failures", "absences"),
        ("Medu", "G_avg"),
        ("Medu", "absences"),
        ("absences", "G_avg")
    ])

    return G_true

def create_true_graph_adult():
    G_true = nx.DiGraph()

    pos = {
        "race": (94, 72),
        "age": (357.660, 9.841),
        "native.country": (591, 77),
        "sex": (808, 77),
        "education": (306, 188),
        "hours.per.week": (591, 259),
        "workclass": (335, 316),
        "marital.status": (493, 383),
        "occupation": (103.380, 447.143),
        "relationship": (686.645, 478.640),
        "income": (451.006, 515.969)
    }

    for node, position in pos.items():
        G_true.add_node(node, pos=position)

    G_true.add_edges_from([
        ("race", "income"),
        ("race", "occupation"),
        ("race", "marital.status"),
        ("race", "hours.per.week"),
        ("race", "education"),
        ("age", "income"),
        ("age", "occupation"),
        ("age", "marital.status"),
        ("age", "education"),
        ("age", "hours.per.week"),
        ("age", "relationship"),
        ("native.country", "education"),
        ("native.country", "workclass"),
        ("native.country", "hours.per.week"),
        ("native.country", "relationship"),
        ("native.country", "income"),
        ("sex", "education"),
        ("sex", "hours.per.week"),
        ("sex", "marital.status"),
        ("sex", "occupation"),
        ("sex", "relationship"),
        ("sex", "income"),
        ("education", "occupation"),
        ("education", "workclass"),
        ("education", "hours.per.week"),
        ("education", "relationship"),
        ("hours.per.week", "workclass"),
        ("hours.per.week", "marital.status"),
        ("hours.per.week", "income"),
        ("marital.status", "relationship"),
        ("marital.status", "occupation"),
        ("marital.status", "income"),
        ("occupation", "income"),
        ("relationship", "income"),
        ("workclass", "occupation"),
        ("workclass", "marital.status")
    ])

    return G_true

def create_true_graph_adult_small():
    G_true = nx.DiGraph()
    
    pos = {
        "age": (339.329, 60),
        "native.country": (852, 277.391),
        "education": (688.439, 55.261),
        "occupation": (79.555, 209.261),
        "hours.per.week": (118.619, 370.870),
        "workclass": (557.290, 210.174),
        "income": (577.781, 440)
    }

    for node, position in pos.items():
        G_true.add_node(node, pos=position)

    G_true.add_edges_from([
        ("age", "income"),
        ("age", "occupation"),
        ("age", "hours.per.week"),
        ("age", "workclass"),
        ("age", "education"),
        ("native.country", "income"),
        ("native.country", "workclass"),
        ("native.country", "hours.per.week"),
        ("native.country", "education"),
        ("education", "occupation"),
        ("education", "hours.per.week"),
        ("education", "workclass"),
        ("education", "income"),
        ("occupation", "income"),
        ("hours.per.week", "income"),
        ("workclass", "income")
    ])

    return G_true

def plot_true_graph(G_true, dataset_name):
    pos = nx.get_node_attributes(G_true, 'pos')

    plt.figure(figsize=(12, 8))
    nx.draw(G_true, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True, arrowstyle='-|>', arrowsize=20)
    plt.title(f"True Causal Graph for {dataset_name.capitalize()} Dataset")
    plt.savefig(f'{dataset_name}_true_graph.png')
    plt.show()

if __name__ == "__main__":
    student_true_graph = create_true_graph_student()
    plot_true_graph(student_true_graph, "student")

    student_small_true_graph = create_true_graph_student_small()
    plot_true_graph(student_small_true_graph, "student_small")

    adult_true_graph = create_true_graph_adult()
    plot_true_graph(adult_true_graph, "adult")

    adult_small_true_graph = create_true_graph_adult_small()
    plot_true_graph(adult_small_true_graph, "adult_small")
