import igraph
import utils
import leidenalg
from sklearn.model_selection import train_test_split

patents = utils.read_patents()
patent_classification = utils.read_patent_classification()
uspatentcitations = utils.read_uspatentcitation()
patents = utils.merge_patents_and_classification(patents, patent_classification)
patents = patents.set_index("id")

delta = 42

range_patents = patents
range_uspatentcitations = uspatentcitations
range_train_patents, range_test_patents = train_test_split(patents)

print("read graph")
g = igraph.Graph().Read_GraphMLz('../Data/graph_with_attributes.xml')

print("read components")
connected_components = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)

print("find subgraphs")
subgraphs = connected_components.subgraphs()
num_subgraphs = len(subgraphs)

range_assigned_patents = utils.igraph_classify_train_test_graph(
    subgraphs, num_subgraphs, range_patents, range_train_patents, range_test_patents
)
