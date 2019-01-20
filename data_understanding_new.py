import utils
import igraph

patents = utils.read_patents()
patent_classification = utils.read_patent_classification()
citations = utils.read_uspatentcitation()
patents = utils.merge_patents_and_classification(patents, patent_classification)


# utils.plot_section_distribution(patents)

# utils.plot_subsection_distribution(patents)

# utils.plot_patents_in_time(patents)

# utils.plot_patent_section_in_time(patents)

print("read graph")
g = igraph.Graph().Read_GraphMLz('../Data/graph_with_attributes.xml')
print("graph stats")
utils.igraph_stats(g)
