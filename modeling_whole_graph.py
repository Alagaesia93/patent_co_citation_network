import igraph
import utils
import leidenalg
import pandas as pd

patents = utils.read_patents()
patent_classification = utils.read_patent_classification()
uspatentcitations = utils.read_uspatentcitation()
patents = utils.merge_patents_and_classification(patents, patent_classification)
patents = patents.set_index("id")

print("read graph")
g = igraph.Graph()
g = utils.add_edges(g, uspatentcitations)
# g = g.as_undirected()

print("read components")
connected_components = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)

print("find subgraphs")
subgraphs = connected_components.subgraphs()
num_subgraphs = len(subgraphs)

print("start to classify")
range_assigned_patents = utils.igraph_classify_whole_graph(
    subgraphs, num_subgraphs, patents
)

# data frame test
forecasted_patents = pd.DataFrame.from_dict(range_assigned_patents, orient='index', columns=['number', 'section_id', 'forecast_section_id'])
utils.write_to_csv(forecasted_patents, 'forecasted_patents')

