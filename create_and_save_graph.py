import igraph
import utils

patents = utils.read_patents()
patent_classification = utils.read_patent_classification()
uspatentcitations = utils.read_uspatentcitation()
patents = utils.merge_patents_and_classification(patents, patent_classification)
patents.set_index("number", inplace=True)

print("create graph")
g = igraph.Graph.TupleList([tuple(x) for x in uspatentcitations.values], directed=True)

len_patents = len(patents)
count = 0
for v in g.vs:
    p = patents.loc[v["name"]]
    print(count / len_patents * 100.0)
    count +=1
    v["section_id"] = p.section_id
    v["subsection_id"] = p.subsection_id
    v["group_id"] = p.group_id
    v["subgroup_id"] = p.subgroup_id

g.write_graphmlz('data/graph_with_attributes.xml')