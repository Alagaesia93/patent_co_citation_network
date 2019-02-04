import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import json
from datetime import datetime
from datetime import timedelta
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import itertools
import igraph
from collections import Counter


def read_and_clean_patent_classification():
    print("Start to read patent classification")
    pat_class = read_original_patent_classification()
    #clean
    print("Start to CLEAN patent classification")
    pat_class = pat_class[pat_class.sequence == 0]
    pat_class = pat_class.drop(['uuid', 'category', 'sequence'], axis=1)
    # add columns
    pat_class = pat_class.assign(forecast_section_id='')
    pat_class = pat_class.assign(forecast_subsection_id='')
    pat_class = pat_class.assign(forecast_group_id='')
    pat_class = pat_class.assign(forecast_subgroup_id='')
    print("FINISHED to read patent classification. Total size", len(pat_class))
    return pat_class


def read_and_clean_patents(pat_class):
    print("Start to read patent")
    pat = read_original_patents()
    # clean
    print("Start to CLEAN patent")
    pat = pat[pat.id == pat.number]
    pat = pat[pat.number.isin(pat_class.patent_id)]
    pat = pat.drop(
        ['type', 'country', 'abstract', 'kind', 'num_claims', 'filename', 'withdrawn'], axis=1
    )
    pat = pat.drop_duplicates()
    print("FINISHED to read patent. Total size", len(pat))
    return pat


def filter_patent_classification(pat_class, pat):
    return pat_class[pat_class.patent_id.isin(pat.number)]


def read_and_clean_original_uspatentcitation(pat):
    print("Start to read patent uspatentcitations")
    cit = read_original_uspatentcitation()
    # clean
    print("Start to CLEAN patent uspatentcitations")
    cit = cit.drop(
        ['uuid', 'date', 'name', 'kind', 'country', 'category', 'sequence'],
        axis=1
    )
    cit = cit.drop_duplicates(subset=['patent_id', 'citation_id'])
    spaces = '           '
    cit.patent_id = cit.patent_id.replace(spaces, '')
    cit.citation_id = cit.citation_id.replace(spaces, '')
    # filter
    cit = cit[cit.patent_id.isin(pat.number)]
    cit = cit[cit.citation_id.isin(pat.number)]
    print("FINISHED to read uspatentcitations. Total size", len(cit))
    return cit


def write_to_csv(df, name):
    print("writing", name)
    df.to_csv('data/'+name+'.tsv', index=False, sep='\t')


def plot_section_distribution(patents, range_start=None, range_end=None, name="plot_section_distribution"):
    if range_start is None:
        range_start = patents['date'].min()
    if range_end is None:
        range_end = patents['date'].max()

    range_patents = patents[patents.date >= range_start]
    range_patents = range_patents[range_patents.date <= range_end]

    patent_section_dict = range_patents.groupby('section_id').size().to_dict()
    labels = patent_section_dict.keys()
    sizes = patent_section_dict.values()

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    # plt.show()
    plt.savefig("fig/"+name+".png",
                transparent=True)
    plt.close()


def plot_subsection_distribution(patents, range_start=None, range_end=None):
    if range_start is None:
        range_start = patents['date'].min()
    if range_end is None:
        range_end = patents['date'].max()

    range_patents = patents[patents.date >= range_start]
    range_patents = range_patents[range_patents.date <= range_end]

    patent_section_dict = range_patents.groupby('section_id').size().to_dict()
    for section in patent_section_dict:
        sel_pat = range_patents[range_patents.section_id == section]
        patent_subsection_dict = sel_pat.groupby('subsection_id').size().to_dict()
        labels = patent_subsection_dict.keys()
        sizes = patent_subsection_dict.values()

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        # plt.show()
        plt.savefig("fig/plot_subsection_distribution_"+section+".png",
                    transparent=True)


def plot_patents_in_time(patents):
    patent_section_dict = patents.groupby(patents.date.dt.year).size().to_dict()
    labels = patent_section_dict.keys()
    sizes = patent_section_dict.values()
    plt.plot(labels, sizes)
    plt.xlabel('Years')
    plt.ylabel('Number of patents')
    plt.title('Patent granted per year')
    # plt.show()
    plt.savefig("fig/plot_patents_in_time.png",
                transparent=True)

    prev_values = 0
    cum_sum = dict()
    for e in patent_section_dict:
        prev_values += patent_section_dict[e]
        cum_sum[e] = prev_values
    plt.plot(labels, cum_sum.values())
    plt.xlabel('Years')
    plt.ylabel('Number of patents')
    plt.title('Cumulative Patent granted per year')
    # plt.show()
    plt.savefig("fig/cumulative_patent_per_year.png",
                transparent=True)


def plot_patent_section_in_time(patents):
    temp = patents.groupby([patents.date.dt.year, 'section_id']).size()
    temp.unstack().plot(kind='bar', stacked=True)
    plt.xlabel('Years')
    plt.ylabel('Number of patents')
    plt.title('Patent section per year')
    # plt.show()
    plt.savefig("fig/patent_section_per_year.png",
                transparent=True)


def read_original_patents():
    print("read original patent file")
    patents = pd.read_csv(
        'data/patent.tsv',
        sep='\t',
        error_bad_lines=False,
        dtype={
            'id': 'object',
            'number': 'object'
        },
    )
    patents.sort_values(by='date', inplace=True)
    print("len(patents)", len(patents))
    return patents


def read_sampled_original_patents(perc=0.5):
    print("read sampled original patent file")
    patents = pd.read_csv(
        'data/patent.tsv',
        sep='\t',
        error_bad_lines=False,
        dtype={
            'id': 'object',
            'number': 'object'
        },
        skiprows=lambda i: i > 0 and random.random() > perc
    )
    patents.sort_values(by='date', inplace=True)
    print("len(patents)", len(patents))
    return patents


def read_original_patent_classification():
    print("read original classification")
    patent_classification = pd.read_csv(
        'data/cpc_current.tsv',
        sep='\t',
        dtype={
            'patent_id': 'object'
        }
    )
    print("len patent_classification", len(patent_classification))
    return patent_classification


def read_sampled_original_patent_classification(patents):
    print("read sampled original classification")
    patent_classification = pd.read_csv(
        'data/cpc_current.tsv',
        sep='\t',
        dtype={
            'patent_id': 'object'
        }
    )
    print("len patent_classification", len(patent_classification))
    print("filtering")
    patent_classification = patent_classification[patent_classification.patent_id.isin(patents.number)]
    print("len patent_classification present in patents", len(patent_classification))
    return patent_classification


def read_original_uspatentcitation():
    print("read original uspatentcitation")
    uspatentcitation = pd.read_csv(
        'data/uspatentcitation.tsv',
        sep='\t',
        error_bad_lines=False,
        dtype={
            'patent_id': 'object',
            'citation_id': 'object'
        },
    )
    print("len(uspatentcitation)", len(uspatentcitation))
    return uspatentcitation


def read_sampled_original_uspatentcitation(patents):
    print("read sampled original uspatentcitation")
    uspatentcitation = pd.read_csv(
        'data/uspatentictation.tsv',
        sep='\t',
        error_bad_lines=False,
        dtype={
            'patent_id': 'object',
            'citation_id': 'object'
        },
    )
    print("len(uspatentcitation)", len(uspatentcitation))
    print("filtering")
    uspatentcitation = uspatentcitation[uspatentcitation.patent_id.isin(patents.number)]
    uspatentcitation = uspatentcitation[uspatentcitation.citation_id.isin(patents.number)]
    print("len(uspatentcitation)", len(uspatentcitation))
    return uspatentcitation


def read_patents():
    print("read patent file")
    patents = pd.read_csv(
        './data/trimmed_patents.tsv',
        sep='\t',
        error_bad_lines=False,
        dtype={
            'id': 'object',
            'number': 'object',
        },
    )
    patents['date'] = pd.to_datetime(patents['date'], format='%Y/%m/%d')
    patents.sort_values(by='date', inplace=True)
    print("len(patents)", len(patents))
    return patents


def read_sampled_patents():
    print("read sampled patent file")
    patents = pd.read_csv(
        './data/trimmed_patents.tsv',
        sep='\t',
        error_bad_lines=False,
        dtype={'id': 'object', 'number': 'object'},
        skiprows=lambda i: i > 0 and random.random() > 0.01
    )
    patents.sort_values(by='date', inplace=True)
    print("len(patents)", len(patents))
    return patents


def read_patent_classification():
    print("read classification")
    patent_classification = pd.read_csv(
        'data/trimmed_cpc_current.tsv',
        sep='\t',
        dtype={'patent_id': 'object'}
    )
    print("len patent_classification", len(patent_classification))
    return patent_classification


def read_sampled_patent_classification(patents):
    print("read classification")
    patent_classification = pd.read_csv(
        'data/trimmed_cpc_current.tsv',
        sep='\t',
        dtype={'patent_id': 'object'}
    )
    print("len patent_classification", len(patent_classification))
    patent_classification = patent_classification[patent_classification.patent_id.isin(patents.number)]
    print("len patent_classification present in patents", len(patent_classification))
    return patent_classification


def read_uspatentcitation():
    print("read uspatentcitation")
    uspatentcitation = pd.read_csv(
        'data/trimmed_uspatentcitations.tsv',
        sep='\t',
        error_bad_lines=False,
        dtype={'patent_id': 'object', 'citation_id': 'object'},
    )
    print("len(uspatentcitation)", len(uspatentcitation))
    return uspatentcitation


def read_sampled_uspatentcitation(patents):
    print("read sampled uspatentcitations")
    uspatentcitation = pd.read_csv(
        'data/trimmed_uspatentcitations.tsv',
        sep='\t',
        error_bad_lines=False,
        dtype={'patent_id': 'object', 'citation_id': 'object'},
    )
    print("len(uspatentcitation)", len(uspatentcitation))
    uspatentcitation = uspatentcitation[uspatentcitation.patent_id.isin(patents.number)]
    uspatentcitation = uspatentcitation[uspatentcitation.citation_id.isin(patents.number)]
    print("len(uspatentcitation)", len(uspatentcitation))
    return uspatentcitation


def merge_patents_and_classification(patents, patent_classification):
    temp = pd.merge(patents, patent_classification, left_on='number', right_on='patent_id', how='inner')
    print("len(merge patents and classification)", len(temp))
    return temp


def create_graph(uspatentcitation):
    g = nx.Graph()
    print("preparing edge tuples")
    edges = [tuple(x) for x in uspatentcitation.values]
    print("ready to add edges")
    g.add_edges_from(edges)
    print("G number of nodes", g.number_of_nodes())
    print("G number of edgeds", g.number_of_edges())
    return g


def connected_components(G):
    cmps = list(nx.connected_components(G))
    print("len cmps", len(cmps))
    return cmps, len(cmps)


def classify_whole_graph(cmps, num_cmps, patents):
    global_assigned_patents = dict()
    count = 1
    patents_length = len(patents)
    print("patents_length", patents_length)
    for cmp in sorted(list(cmps), key=len, reverse=True):
        print(str(count) + " out of " + str(num_cmps))
        print("percentage:", count / num_cmps * 100)
        print("component length", len(cmp))
        print("relative component length", len(cmp) / patents_length * 100)
        cmp_patents = patents[patents.number.isin(cmp)]
        print("", len(cmp_patents))
        prevision = {
            'section_id': cmp_patents.groupby('section_id').size().idxmax(),
            'subsection_id': cmp_patents.groupby('subsection_id').size().idxmax(),
            'group_id': cmp_patents.groupby('group_id').size().idxmax(),
            'subgroup_id': cmp_patents.groupby('subgroup_id').size().idxmax(),
        }
        print("prevision", prevision)
        for row in cmp_patents.number.tolist():
            global_assigned_patents[row] = {
                'forecast_section_id': prevision['section_id'],
                'forecast_subsection_id': prevision['subsection_id'],
                'forecast_group_id': prevision['group_id'],
                'forecast_subgroup_id': prevision['subgroup_id']
            }
        print("_________________________________")
        count += 1

    sections = sorted(patents['section_id'].drop_duplicates())
    subsections = sorted(patents['subsection_id'].drop_duplicates())
    # groups = sorted(patents['group_id'].drop_duplicates())
    # subgroups = sorted(patents['subgroup_id'].drop_duplicates())
    evaluate(global_assigned_patents, patents, sections, subsections)
    return global_assigned_patents


def classify_train_test_graph(cmps, num_cmps, range_patents, range_train_patents, range_test_patents):
    global_assigned_patents = dict()
    count = 1
    patents_length = len(range_patents)
    print("patents_length", patents_length)
    print("check dates")
    print("min date range patents", range_patents['date'].min())
    print("max date range patents", range_patents['date'].max())
    print("min date range train patents", range_train_patents['date'].min())
    print("max date range train patents", range_train_patents['date'].max())
    print("min date range test patents", range_test_patents['date'].min())
    print("max date range test patents", range_test_patents['date'].max())
    for cmp in sorted(list(cmps), key=len, reverse=True):
        print(str(count) + " out of " + str(num_cmps))
        print("percentage:", count / num_cmps * 100)
        print("component length", len(cmp))
        print("relative component length", len(cmp) / patents_length * 100)
        cmp_patents = range_patents[range_patents.number.isin(cmp)]
        print("len cmp_patents", len(cmp_patents))
        cmp_train_patents = range_train_patents[range_train_patents.number.isin(cmp)]
        print("len cmp train patents", len(cmp_train_patents))
        cmp_test_patents = range_test_patents[range_test_patents.number.isin(cmp)]
        print("len cmp test patents", len(cmp_test_patents))
        if len(cmp_train_patents) > 0:
            prevision = {
                'section_id': cmp_train_patents.groupby('section_id').size().idxmax(),
                'subsection_id': cmp_train_patents.groupby('subsection_id').size().idxmax(),
                'group_id': cmp_train_patents.groupby('group_id').size().idxmax(),
                'subgroup_id': cmp_train_patents.groupby('subgroup_id').size().idxmax(),
            }
            print("prevision", prevision)
            for row in cmp_test_patents.number.tolist():
                global_assigned_patents[row] = {
                    'forecast_section_id': prevision['section_id'],
                    'forecast_subsection_id': prevision['subsection_id'],
                    'forecast_group_id': prevision['group_id'],
                    'forecast_subgroup_id': prevision['subgroup_id']
                }
        print("_________________________________")
        count += 1
    sections = sorted(range_patents['section_id'].drop_duplicates())
    subsections = sorted(range_patents['subsection_id'].drop_duplicates())
    # groups = sorted(patents['group_id'].drop_duplicates())
    # subgroups = sorted(patents['subgroup_id'].drop_duplicates())
    evaluate(global_assigned_patents, range_patents, sections, subsections)
    return global_assigned_patents


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          range_dates=[]):
    start_date = range_dates[0]
    end_date = range_dates[1]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix from "+str(start_date)+" to "+str(end_date))
    else:
        print("Confusion matrix, without normalization "+str(start_date)+" to "+str(end_date))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if normalize:
        plt.savefig("fig/normalized_confusion_matrix_from_"+str(start_date)+"_to_"+str(end_date)+".png", transparent=True)
    else:
        plt.savefig("fig/confusion_matrix_from_" + str(start_date) + "_to_" + str(end_date) + ".png", transparent=True)


def evaluate(global_assigned_patents, patents, sections, subsections):
    range_start = patents['date'].min()
    range_end = patents['date'].max()
    print("Starting to evaluate")
    section_y_true = patents[
         patents.number.isin(global_assigned_patents.keys())
    ].sort_values(by='number')['section_id']
    section_y_pred = list(
        map(
            lambda x: x[1]['forecast_section_id'],
            list(sorted(global_assigned_patents.items()))
        )
    )
    print("section classification report", classification_report(section_y_true, section_y_pred, target_names=sections))
    section_cnf_matrix = confusion_matrix(section_y_true, section_y_pred, labels=sections)
    plt.figure()
    plot_confusion_matrix(section_cnf_matrix, classes=sections, title='Section Confusion Matrix', range_dates=[range_start, range_end])
    plt.figure()
    plot_confusion_matrix(section_cnf_matrix, classes=sections, title='Section Confusion Matrix with normalization', normalize=True, range_dates=[range_start, range_end])
    # plt.show()

    print("Starting to evaluate subsections")
    subsection_y_true = patents[
        patents.number.isin(global_assigned_patents.keys())
    ].sort_values(by='number')['subsection_id']
    subsection_y_pred = list(
        map(
            lambda x: x[1]['forecast_subsection_id'],
            list(sorted(global_assigned_patents.items()))
        )
    )
    print("subsections classification report", classification_report(subsection_y_true, subsection_y_pred, target_names=subsections))


def new_evaluate(global_assigned_patents, patents, sections):
    range_start = patents['date'].min()
    range_end = patents['date'].max()
    print("Starting to evaluate")
    forecasted_patents = pd.DataFrame.from_dict(global_assigned_patents, orient='index', columns=['number', 'section_id', 'forecast_section_id'])
    forecasted_patents = forecasted_patents[~forecasted_patents.forecast_section_id.isnull()]
    section_y_true = forecasted_patents['section_id']
    section_y_pred = forecasted_patents['forecast_section_id']
    print("section classification report", classification_report(section_y_true, section_y_pred, target_names=sections))


def classify_section(selected_patents, uspatentcitations):
    print("len(selected_patents", len(selected_patents))
    uspatentcitations = uspatentcitations[uspatentcitations.patent_id.isin(selected_patents.number)]
    print("len(uspatentcitations)", len(uspatentcitations))
    uspatentcitations = uspatentcitations[uspatentcitations.citation_id.isin(selected_patents.number)]
    print("len(uspatentcitations)", len(uspatentcitations))
    G = create_graph(uspatentcitations)
    subgraph_connected_components, num_cmps = connected_components(G)
    print(num_cmps)
    classify_whole_graph(subgraph_connected_components, num_cmps, selected_patents)


def find_range_dataframes(my_range, patents, uspatentcitations):
    print("looking for range patents and citations")
    range_patents = patents[patents.date >= my_range.range_start]
    range_patents = range_patents[range_patents.date <= my_range.range_end]
    print("len(range_patents)", len(range_patents))
    range_train_patents = range_patents[range_patents.date < my_range.train_end]
    print("len(range_train_patents)", len(range_train_patents))
    range_test_patents = range_patents[range_patents.date >= my_range.train_end]
    print("len(range_test_patents)", len(range_test_patents))
    range_uspatentcitations = uspatentcitations[
        uspatentcitations.patent_id.isin(range_patents.number)
    ]
    range_uspatentcitations = range_uspatentcitations[
        range_uspatentcitations.citation_id.isin(range_patents.number)
    ]
    print("len(range_uspatentcitations)", len(range_uspatentcitations))
    return range_patents, range_train_patents, range_test_patents, range_uspatentcitations


def find_range_dataframes_from_beginning(my_range, patents, uspatentcitations):
    print("looking for range patents and citations")
    range_patents = patents[patents.date >= my_range.range_start]
    range_patents = range_patents[range_patents.date <= my_range.range_end]
    print("len(range_patents)", len(range_patents))
    range_train_patents = range_patents[range_patents.date < my_range.train_end]
    print("len(range_train_patents)", len(range_train_patents))
    range_test_patents = range_patents[range_patents.date >= my_range.train_end]
    print("len(range_test_patents)", len(range_test_patents))
    patents_from_beginning = patents[patents.date <= my_range.range_end]
    print("len(patents_from_beginning)", len(patents_from_beginning))
    range_uspatentcitations = uspatentcitations[
        uspatentcitations.patent_id.isin(patents_from_beginning.number)
    ]
    range_uspatentcitations = range_uspatentcitations[
        range_uspatentcitations.citation_id.isin(patents_from_beginning.number)
    ]
    print("len(range_uspatentcitations)", len(range_uspatentcitations))
    return range_patents, range_train_patents, range_test_patents, range_uspatentcitations


def add_edges(G, range_uspatentcitations):
    print("G number of nodes before", G.vcount())
    print("G number of edges before", G.ecount())
    nodes = list(
        set(
            set(range_uspatentcitations.patent_id).union(
                set(range_uspatentcitations.citation_id)
            )
        )
    )
    # first node
    G.add_vertices(nodes)
    print("done adding vertices")
    G.add_edges([tuple(x) for x in range_uspatentcitations.values])
    print("G number of nodes after", G.vcount())
    print("G number of edges after", G.ecount())
    return G


def igraph_classify_train_test_graph(subgraphs, num_subgraphs, range_patents, range_train_patents, range_test_patents):
    global_assigned_patents = range_patents[['number', 'section_id']].to_dict('index')
    count = 1
    patents_length = len(range_patents)
    print("patents_length", patents_length)
    print("check dates")
    print("min date range patents", range_patents['date'].min())
    print("max date range patents", range_patents['date'].max())
    print("min date range train patents", range_train_patents['date'].min())
    print("max date range train patents", range_train_patents['date'].max())
    print("min date range test patents", range_test_patents['date'].min())
    print("max date range test patents", range_test_patents['date'].max())
    for sbg in subgraphs:
        print(str(count) + " out of " + str(num_subgraphs))
        print("percentage:", count / num_subgraphs * 100)
        print("component length", sbg.vcount())
        print("relative component length", sbg.vcount() / patents_length * 100)
        cmp_patents = range_patents[range_patents.number.isin(sbg.vs["name"])]
        print("len cmp_patents", len(cmp_patents))
        cmp_train_patents = range_train_patents[range_train_patents.number.isin(sbg.vs["name"])]
        print("len cmp train patents", len(cmp_train_patents))
        cmp_test_patents = range_test_patents[range_test_patents.number.isin(sbg.vs["name"])]
        print("len cmp test patents", len(cmp_test_patents))
        if len(cmp_train_patents) > 0:
            prevision = cmp_train_patents.groupby('section_id').size().idxmax()
            print("prevision", prevision)
            sbg.vs["forecast_section_id"] = prevision
            print("forecast section id", sbg.vs["forecast_section_id"])
            # for row in cmp_test_patents.number.tolist():
            #     global_assigned_patents[row]['forecast_section_id'] = prevision
            if count <= 10:
                plot_subgraph(sbg, name="prop_"+str(count), selection="section_id")
                plot_subgraph(sbg, name="prop_" + str(count), selection="forecast_section_id")
        print("_________________________________")
        count += 1
    sections = sorted(range_patents['section_id'].drop_duplicates())
    # subsections = sorted(range_patents['subsection_id'].drop_duplicates())
    # groups = sorted(patents['group_id'].drop_duplicates())
    # subgroups = sorted(patents['subgroup_id'].drop_duplicates())
    # evaluate(global_assigned_patents, range_patents, sections, subsections)
    new_evaluate(global_assigned_patents, range_patents, sections)
    return global_assigned_patents


def igraph_stats(g):
    igraph_stats_directed(g)
    igraph_stats_undirected(g)


def igraph_stats_directed(g):
    print(igraph.summary(g))
    print("number of nodes", g.vcount())
    print("number of edges", g.ecount())
    print("connected components")
    components = g.components()
    print("find subgraphs")
    subgraphs = components.subgraphs()
    print("number of components", len(subgraphs))
    print("giant component size", components.giant().vcount())
    in_degree = g.indegree()
    out_degree = g.indegree()
    degree = g.degree()
    nodes_degree = dict()
    nodes_in_degree = dict()
    nodes_out_degree = dict()
    index = 0
    for node in g.vs:
        nodes_degree[node["name"]] = degree[index]
        nodes_in_degree[node["name"]] = in_degree[index]
        nodes_out_degree[node["name"]] = out_degree[index]
        index += 1
    nodes_in_degree = sorted(nodes_in_degree.items(), key=lambda kv: kv[1], reverse=True)
    print("top 100 sorted nodes_in_degree", nodes_in_degree[:100])
    nodes_out_degree = sorted(nodes_out_degree.items(), key=lambda kv: kv[1], reverse=True)
    print("top 100 sorted nodes_out_degree", nodes_out_degree[:100])
    nodes_degree = sorted(nodes_degree.items(), key=lambda kv: kv[1], reverse=True)
    print("top 100 sorted nodes_degree", nodes_degree[:100])
    plot_degree_distribution(in_degree, "nodes_in_degree")
    plot_degree_distribution(out_degree, "nodes_out_degree")
    plot_degree_distribution(degree, "nodes degree")
    print("degree distribution", g.degree_distribution())
    print("average_path_length", g.average_path_length())


def igraph_stats_undirected(g):
    print("undirected")
    und = g.as_undirected()
    print(igraph.summary(und))
    components = und.components()
    subgraphs = components.subgraphs()
    print("number of components", len(subgraphs))
    print("giant component size", components.giant().vcount())


def plot_degree_distribution(degrees, name):
    counter = Counter(degrees)
    print(counter)
    plt.scatter(counter.keys(), counter.values())
    # plt.show()
    plt.savefig("fig/"+name+".png",
                transparent=True)


def igraph_classify_whole_graph(subgraphs, num_subgraphs, patents):
    global_assigned_patents = patents[['number', 'section_id']].to_dict('index')
    count = 1
    patents_length = len(patents)
    print("patents_length", patents_length)
    print("check dates")
    for sbg in subgraphs:
        print(str(count) + " out of " + str(num_subgraphs))
        print("percentage:", count / num_subgraphs * 100)
        print("component length", sbg.vcount())
        print("relative component length", sbg.vcount() / patents_length * 100)
        cmp_patents = patents[patents.number.isin(sbg.vs["name"])]
        print("len cmp_patents", len(cmp_patents))
        prevision = cmp_patents.groupby('section_id').size().idxmax()
        print("prevision", prevision)
        for row in cmp_patents.number.tolist():
            global_assigned_patents[row]['forecast_section_id'] = prevision
        print("_________________________________")
        count += 1
    # sections = sorted(patents['section_id'].drop_duplicates())
    # subsections = sorted(patents['subsection_id'].drop_duplicates())
    # evaluate(global_assigned_patents, patents, sections, subsections)
    return global_assigned_patents


def plot_subgraph(sbg, name="test", selection="section_id", folder="test"):
    print("plot subgraph", igraph.summary(sbg))
    color_dict = {
        "A": "lime",
        "B": "red",
        "C": "gold",
        "D": "darkgreen",
        "E": "slategray",
        "F": "blue",
        "G": "black",
        "H": "white"
    }
    sbg.vs["color"] = [color_dict[section_id] for section_id in sbg.vs[selection]]
    print("set layout")
    # layout = sbg.layout("drl")
    print("plot")
    # igraph.plot(sbg, 'fig/'+name+'.png', layout=layout)
    igraph.plot(sbg, 'fig/' + folder + '/' + name + '.png')


class Range:
    min_date = ''
    # min_date_datetime = ''
    max_date = ''
    # max_date_datetime = ''
    range_start = ''
    # range_start_datetime = ''
    range_end = ''
    # range_end_datetime = ''
    train_end = ''
    # train_end_datetime = ''
    total_years = ''
    previous_range_end = ''

    def __init__(self, delta, train_percentage, min_date, max_date):
        test_percentage = 1 - train_percentage
        self.min_date = min_date.replace(day=1, month=1)

        self.max_date = max_date.replace(day=31, month=12)

        self.range_start = self.min_date

        self.range_end = self.range_start.replace(year=(self.range_start.year + delta))

        self.total_years = self.range_end.year - self.range_start.year

        self.train_end = self.range_end.replace(
            year=(self.range_end.year - round(test_percentage * delta))
        )

    def increase(self, delta, train_percentage):
        test_percentage = 1 - train_percentage

        self.previous_range_end = self.range_end

        self.range_start = self.range_end + timedelta(days=1)

        self.range_end = self.range_start.replace(
            year=(self.range_start.year + delta)
        )

        if self.max_date.year - self.range_end.year < delta:
            print(
                "We have only " +
                str(self.max_date.year - self.range_end.year) +
                " years left. Let's extend the range until the very end"
            )
            self.range_end = self.max_date

        self.train_end = self.range_end.replace(
            year=(self.range_end.year - round(test_percentage * self.total_years)))

    def print(self):
        print("min_date", self.min_date)
        print("max_date", self.max_date)
        print("range_start", self.range_start)
        print("range_end", self.range_end)
        print("total_years", self.total_years)
        print("train_end", self.train_end)

    def increase_proportionally(self, delta, train_percentage):
        train_add_years = 6

        test_percentage = 1 - train_percentage

        self.range_end = self.range_end.replace(
            year=(self.range_end.year + delta)
        )

        self.range_start = self.range_start.replace(
            year=(self.range_start.year + train_add_years)
        )

        if self.max_date.year - self.range_end.year < delta:
            print(
                "We have only " +
                str(self.max_date.year - self.range_end.year) +
                " years left. Let's extend the range until the very end"
            )
            self.range_end = self.max_date

        self.train_end = self.range_end.replace(
            year=(self.range_end.year - int(test_percentage * self.total_years)))


