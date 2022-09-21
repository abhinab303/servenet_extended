from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import NeighborLoader
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
# g_data = Data(pyg_graph)
# print(pyg_graph)

loader = NeighborLoader(pyg_graph,
                        num_neighbors=[5]*pyg_graph.num_nodes,
                        batch_size=128)


class MyOwnDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)




pass
# for batch in loader:
#     print(batch)
#     print(batch.num_graphs)

