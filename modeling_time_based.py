import utils
import igraph
import leidenalg

patents = utils.read_patents()
patent_classification = utils.read_patent_classification()
uspatentcitations = utils.read_uspatentcitation()
patents = utils.merge_patents_and_classification(patents, patent_classification)

delta = 10
train_percentage = 0.8

my_range = utils.Range(delta, train_percentage, patents['date'].min(), patents['date'].max())
my_range.print()
g = igraph.Graph(directed=True)
global_assigned_patents = dict()

print("start of while")
while my_range.range_end <= my_range.max_date:
    range_patents, range_train_patents, range_test_patents, range_uspatentcitations = utils.find_range_dataframes(
        my_range, patents, uspatentcitations
    )
    g = utils.add_edges(g, range_uspatentcitations)
    print("finding components")
    connected_components = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    subgraphs = connected_components.subgraphs()
    num_subgraphs = len(subgraphs)
    range_assigned_patents = utils.igraph_classify_train_test_graph(
        subgraphs, num_subgraphs, range_patents, range_train_patents, range_test_patents
    )
    for key in range_assigned_patents.keys():
        global_assigned_patents[key] = range_assigned_patents[key]
    my_range.increase(delta, train_percentage)
    my_range.print()


print("Final evaluation")
sections = sorted(patents['section_id'].drop_duplicates())
subsections = sorted(patents['subsection_id'].drop_duplicates())
utils.evaluate(global_assigned_patents, patents, sections, subsections)

