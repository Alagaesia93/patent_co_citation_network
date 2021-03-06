import igraph
import utils
import leidenalg
from sklearn.model_selection import train_test_split

patents = utils.read_patents()
patent_classification = utils.read_patent_classification()
uspatentcitations = utils.read_uspatentcitation()
patents = utils.merge_patents_and_classification(patents, patent_classification)
patents = patents.set_index("id")
len_patents = len(patents)

print("range variables")
delta = 10
train_percentage = 0.8
my_range = utils.Range(delta, train_percentage, patents['date'].min(), patents['date'].max())
my_range.print()
g = igraph.Graph(directed=True)
index = 0

print("start of while")
while my_range.range_end <= my_range.max_date:
    print("range patents")
    range_patents, range_train_patents, range_test_patents, range_uspatentcitations = utils.find_range_dataframes(
        my_range, patents, uspatentcitations
    )

    print("add edges")
    g = utils.add_edges(g, range_uspatentcitations)

    count = 0
    print("add section id")
    for v in g.vs:
        p = patents.loc[v["name"]]
        # print(count / len_patents * 100.0)
        v["section_id"] = p.section_id
        count += 1

    print("read components")
    connected_components = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)

    print("find subgraphs")
    subgraphs = connected_components.subgraphs()
    num_subgraphs = len(subgraphs)

    for sbg in subgraphs[0:10]:
        print("train test")
        cmp_patents = range_patents[range_patents.number.isin(sbg.vs["name"])]
        print("len cmp_patents", len(cmp_patents))
        cmp_train_patents = range_train_patents[range_train_patents.number.isin(sbg.vs["name"])]
        print("len cmp train patents", len(cmp_train_patents))
        cmp_test_patents = range_test_patents[range_test_patents.number.isin(sbg.vs["name"])]
        print("len cmp test patents", len(cmp_test_patents))

        print("prevision")
        prevision = cmp_train_patents.groupby('section_id').size().idxmax()
        sbg.vs["forecast_section_id"] = prevision

        print("create i-th graph", index)
        sbg.write_graphmlz('data/graphs/prop_ranges/section_'+str(index)+'.xml')

        print("plot component")
        utils.plot_subgraph(sbg, name="prop_" + str(index), selection="section_id", folder="prop_ranges")
        # utils.plot_subgraph(sbg, name="prop_" + str(index)+'_forecast', selection="forecast_section_id", folder="time_based")
        index += 1

    my_range.increase_proportionally(delta, train_percentage)
    my_range.print()
    G = igraph.Graph(directed=True)  # restore the initial graph

