import igraph
import utils
import leidenalg

print("read graph")
g = igraph.Graph().Read_GraphMLz('../Data/graph_with_attributes.xml')

# print("read components")
# connected_components = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)

utils.plot_subgraph(g, name="whole_graph")