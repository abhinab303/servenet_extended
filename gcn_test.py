from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
import pickle

graph_path = "/home/aa7514/PycharmProjects/servenet_extended/files/graph.pickle"
feature_path = "/home/aa7514/PycharmProjects/servenet_extended/files/word_embeddings.pickle"

with open(graph_path, "rb") as f:
    graph = pickle.load(f)

with open(feature_path, "rb") as f:
    feature_matrix = pickle.load(f)

# print(graph[15000])

graph_copy = graph.copy()

for node_index, node in graph_copy.nodes(data=True):
    try:
        node["feature"] = feature_matrix[node_index]
    except Exception as ex:
        continue


pyg_graph = from_networkx(graph)

# print(pyg_graph)

loader = DataLoader(pyg_graph, batch_size=1)

pass
# for batch in loader:
#     print(batch)
#     print(batch.num_graphs)

