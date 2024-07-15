from causallearn.search.FCMBased import lingam
from causallearn.search.FCMBased.lingam.utils import make_dot

def run_lingam_algorithm(data, labels):
    model_lingam = lingam.ICALiNGAM()
    model_lingam.fit(data)
    digraph = make_dot(model_lingam.adjacency_matrix_, labels=labels)
    digraph.render('lingam_graph', format='png')

    graph_dot_string = digraph.source
    return graph_dot_string

if __name__ == "__main__":
    from data_preparation import load_and_prepare_data
    file_path = 'data/student-por_raw.csv'
    df_encoded, labels, data = load_and_prepare_data(file_path)
    graph_dot_string = run_lingam_algorithm(data, labels)
    print("LiNGAM Algorithm graph created.")