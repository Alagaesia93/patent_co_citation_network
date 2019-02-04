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

print("range patents")
range_patents = patents
range_uspatentcitations = uspatentcitations
range_train_patents, range_test_patents = train_test_split(patents)

print("create graph")
g = igraph.Graph.TupleList([tuple(x) for x in uspatentcitations.values], directed=True)
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
index = 0
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
    sbg.write_graphmlz('data/graphs/whole_graph/section_'+str(index)+'.xml')

    print("plot component")
    utils.plot_subgraph(sbg, name="prop_" + str(index), selection="section_id", folder="whole_graph")
    # utils.plot_subgraph(sbg, name="prop_" + str(index)+'_forecast', selection="forecast_section_id", folder="whole_graph")

    index += 1

