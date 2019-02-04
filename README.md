# COMMUNITY DETECTION IN PATENT CO-CITATION NETWORK

Patents have always been closely related to innovation in the world. The number of applications and grants keeps growing and patents are more and more a fundamental part of the competitive advantage for a business. 

It is important and necessary to categorize and organize this amount of data to let interested people find what they need. In more than 300 years of patents several classification methods have been used, but they always require a domain expert who understands and proposes a class. 

Our goal is to use patent co-citation network to automatically classify a patent in the correct categories, by using graphs and community discovery algorithms like Louvain and Leiden. 

The built graph has more than 6 million nodes and over 70 million edges, from PatentsView.org dataset. It was built using igraph library after a deep cleaning of the original dataset. The patents span from 1976 to 2018. 

Four models have been built: a feasibility study with all the graph, a train test split of the nodes, a time-based split adding to the existing graph and a time-based split that eliminates the eldest edges. 

The results are promising, with about 60% of accuracy and recall at section level, using ONLY the co-citation network, and nothing else. We can’t infer that one the four proposed model is far better than the others.

Our model can be improved using text-based models on string attributes like title and description, or using unsupervised learning to develop a new classification model

# Installation

Python 3.7

Igraph and Leiden libraries

Data from PatentsView.org
