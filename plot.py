import igraph
import utils
import leidenalg
from sklearn.model_selection import train_test_split

# patents = utils.read_patents()
# patent_classification = utils.read_patent_classification()
# uspatentcitations = utils.read_uspatentcitation()
# patents = utils.merge_patents_and_classification(patents, patent_classification)
# patents = patents.set_index("id")

# range_patents = patents
# range_uspatentcitations = uspatentcitations
# range_train_patents, range_test_patents = train_test_split(patents)

print("read graph")
g = igraph.Graph().Read_GraphMLz('../Data/graph_with_attributes.xml')

print("read components")
connected_components = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
#
print("find subgraphs")
subgraphs = connected_components.subgraphs()
# sbg = subgraphs[0]
# sbg.write_graphmlz('data/subgraph.xml')
num_subgraphs = len(subgraphs)
index = 0
for sbg in subgraphs[0:10]:
    print(index)
    # cmp_patents = patents[patents.number.isin(sbg.vs["name"])]
    # utils.plot_section_distribution(cmp_patents, name="train_test_"+str(index))
    utils.plot_subgraph(sbg, name="train_test_"+str(index))
    index += 1

# -------------- train test split

# delta = 10
# train_percentage = 0.8
# index = 0
#
# my_range = utils.Range(delta, train_percentage, patents['date'].min(), patents['date'].max())
# my_range.print()
# g = igraph.Graph(directed=True)
# global_assigned_patents = dict()
#
# print("start of while")
# while my_range.range_end <= my_range.max_date:
#     range_patents, range_train_patents, range_test_patents, range_uspatentcitations = utils.find_range_dataframes_from_beginning(
#         my_range, patents, uspatentcitations
#     )
#     g = utils.add_edges(g, range_uspatentcitations)
#     print("finding components")
#     connected_components = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
#     subgraphs = connected_components.subgraphs()
#     num_subgraphs = len(subgraphs)
#
#     for sbg in subgraphs[0:10]:
#         print(index)
#         # sbg.write_graphmlz('data/graph_with_attributes_'+str(index)+'.xml')
#         index += 1
#         cmp_patents = range_patents[range_patents.number.isin(sbg.vs["name"])]
#         utils.plot_section_distribution(cmp_patents, name="modeling_whole_time_based_" + str(index))
#
#     my_range.increase(delta, train_percentage)
#     my_range.print()
#     G = igraph.Graph(directed=True)


# --------------- time proportionally

# delta = 10
# train_percentage = 0.8
# index = 0
#
#
# my_range = utils.Range(delta, train_percentage, patents['date'].min(), patents['date'].max())
# my_range.print()
# G = igraph.Graph(directed=True)
#
# global_assigned_patents = dict()
#
#
# while my_range.range_end <= my_range.max_date:
#     range_patents, range_train_patents, range_test_patents, range_uspatentcitations = utils.find_range_dataframes(
#         my_range, patents, uspatentcitations
#     )
#     G = utils.add_edges(G, range_uspatentcitations)
#     print("finding components")
#     connected_components = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
#     subgraphs = connected_components.subgraphs()
#     num_subgraphs = len(subgraphs)
#
#     for sbg in subgraphs[0:10]:
#         print(index)
#         # sbg.write_graphmlz('data/graph_with_attributes_'+str(index)+'.xml')
#         index += 1
#         cmp_patents = range_patents[range_patents.number.isin(sbg.vs["name"])]
#         utils.plot_section_distribution(cmp_patents, name="modeling_proportional_time_based_" + str(index))
#
#     my_range.increase_proportionally(delta, train_percentage)
#     my_range.print()
#     G = igraph.Graph(directed=True) # restore the initial graph
#
