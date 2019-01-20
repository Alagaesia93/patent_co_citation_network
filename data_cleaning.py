import utils

patent_classification = utils.read_and_clean_patent_classification()
patents = utils.read_and_clean_patents(patent_classification)
# filter patent_classification again
patent_classification = utils.filter_patent_classification(patent_classification, patents)

citations = utils.read_and_clean_original_uspatentcitation(patents)

utils.write_to_csv(patents, 'trimmed_patents')
utils.write_to_csv(patent_classification, 'trimmed_cpc_current')
utils.write_to_csv(citations, 'trimmed_uspatentcitations')

