
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import time
import datetime
import sys
import networkx as nx

n_processors = 20
import os


def make_parallel(fn, x, num_tasks):
    start = time.time()
    len_split = int(len(x)/num_tasks)
    indices = [[i*len_split, (i+1)*len_split] for i in range(num_tasks)]
    indices[-1][-1] = len(x)

    x_split = [x[t[0]:t[1]] for t in indices]
    
    res = []
    with Pool(num_tasks) as p:
        for i, t in enumerate(p.imap(fn, x_split, 1)):
            #print(t)
            res += t
            sys.stderr.write('\rdone {:%}, remaining: {}'.format(
                (i+1)/num_tasks, num_tasks - (i+1)
            ))
            sys.stderr.flush()
        #res = p.imap_unordered(to_edgelists, data_split)
    print('')
    print(time.time() - start)
    return res


def build_comment_author_table(indices):
    dict_tuple = {}
    for i in indices:
        comment_id = data.iloc[i]['id']
        author = data.iloc[i]['author']
        dict_tuple[comment_id] = author
    return list(dict_tuple.items())

with open('edgelist_large', 'r') as f:
    edges = [x.split() for x in f.readlines()]
graph_only_comments = nx.MultiDiGraph()


edge_keys = graph_only_comments.add_edges_from(edges)


def get_average_out_neighbors_in_degrees(ids):
    out_neighbors = [list(set([t[1] for t in graph_only_comments.out_edges(x)])) for x in ids]
    out_nbrs_in_degrees = [
        list(dict(graph_only_comments.in_degree(x)).values()) for x in out_neighbors
    ]
    out_nbrs_in_degrees_mean = np.array([np.mean(x) if len(x) > 0 else 0 for x in out_nbrs_in_degrees])
    return list(out_nbrs_in_degrees_mean)

def get_average_out_neighbors_out_degrees(ids):
    out_neighbors = [list(set([t[1] for t in graph_only_comments.out_edges(x)])) for x in ids]
    out_nbrs_out_degrees = [
        list(dict(graph_only_comments.out_degree(x)).values()) for x in out_neighbors
    ]
    out_nbrs_out_degrees_mean = np.array([np.mean(x) if len(x) > 0 else 0 for x in out_nbrs_out_degrees])
    return list(out_nbrs_out_degrees_mean)

def get_average_in_neighbors_in_degrees(ids):
    in_neighbors = [list(set([t[0] for t in graph_only_comments.in_edges(x)])) for x in ids]
    in_nbrs_in_degrees = [
        list(dict(graph_only_comments.in_degree(x)).values()) for x in in_neighbors
    ]
    in_nbrs_in_degrees_mean = np.array([np.mean(x) if len(x) > 0 else 0 for x in in_nbrs_in_degrees])
    return list(in_nbrs_in_degrees_mean)


def get_average_in_neighbors_out_degrees(ids):
    in_neighbors = [list(set([t[0] for t in graph_only_comments.in_edges(x)])) for x in ids]
    in_nbrs_in_degrees = [
        list(dict(graph_only_comments.in_degree(x)).values()) for x in in_neighbors
    ]
    in_nbrs_in_degrees_mean = np.array([np.mean(x) if len(x) > 0 else 0 for x in in_nbrs_in_degrees])
    return list(in_nbrs_in_degrees_mean)



def get_average_neighbors_degrees(ids):
    neighbors = [set(list(set([t[1] for t in graph_only_comments.out_edges(x)])) + 
                    list(set([t[1] for t in graph_only_comments.out_edges(x)]))) for x in ids]
    nbrs_degrees = [
        list(dict(graph_only_comments.degree(x)).values()) for x in neighbors
    ]
    nbrs_degrees_mean = np.array([np.mean(x) if len(x) > 0 else 0 for x in nbrs_degrees])
    return list(nbrs_degrees_mean)

out_nbrs_in_degrees_mean = np.array(make_parallel(get_average_out_neighbors_in_degrees, 
                                                 list(graph_only_comments.nodes()), n_processors))

np.savetxt('out_nbrs_in_degrees_mean', out_nbrs_in_degrees_mean)
out_nbrs_out_degrees_mean = np.array(make_parallel(get_average_out_neighbors_out_degrees, 
                                                 list(graph_only_comments.nodes()), n_processors))
np.savetxt('out_nbrs_out_degrees_mean', out_nbrs_out_degrees_mean)

in_nbrs_out_degrees_mean = np.array(make_parallel(get_average_in_neighbors_out_degrees, 
                                                 list(graph_only_comments.nodes()), n_processors))
np.savetxt('in_nbrs_out_degrees_mean', in_nbrs_out_degrees_mean)
in_nbrs_in_degrees_mean = np.array(make_parallel(get_average_in_neighbors_in_degrees, 
                                                 list(graph_only_comments.nodes()), n_processors))
np.savetxt('in_nbrs_in_degrees_mean', in_nbrs_out_degrees_mean)


nbrs_degrees_mean = np.array(make_parallel(get_average_neighbors_degrees, 
                                                 list(graph_only_comments.nodes()), n_processors))


print('{:.2f}% have neighbors average degrees larger than 0, and {:.2f}% larger than 1'.format(
        100.0*np.sum(nbrs_degrees_mean>0) / len(nbrs_degrees_mean),
        100.0*np.sum(nbrs_degrees_mean>1) / len(nbrs_degrees_mean))
)


print('{:.2f}% have out neighbors average in degrees larger than 0, and {:.2f}% larger than 1'.format(
        100.0*np.sum(out_nbrs_in_degrees_mean>0) / len(out_nbrs_in_degrees_mean),
        100.0*np.sum(out_nbrs_in_degrees_mean>1) / len(out_nbrs_in_degrees_mean))
)


print('{:.2f}% have out neighbors average out degrees larger than 0, and {:.2f}% larger than 1'.format(
        100.0*np.sum(out_nbrs_out_degrees_mean>0) / len(out_nbrs_out_degrees_mean),
        100.0*np.sum(out_nbrs_out_degrees_mean>1) / len(out_nbrs_out_degrees_mean))
)


print('{:.2f}% have in neighbors average out degrees larger than 0, and {:.2f}% larger than 1'.format(
        100.0*np.sum(in_nbrs_out_degrees_mean>0) / len(in_nbrs_out_degrees_mean),
        100.0*np.sum(in_nbrs_out_degrees_mean>1) / len(in_nbrs_out_degrees_mean))
)


print('{:.2f}% have in neighbors average in degrees larger than 0, and {:.2f}% larger than 1'.format(
        100.0*np.sum(in_nbrs_in_degrees_mean>0) / len(in_nbrs_in_degrees_mean),
        100.0*np.sum(in_nbrs_in_degrees_mean>1) / len(in_nbrs_in_degrees_mean))
)

graph_only_comments_simple = nx.Graph(graph_only_comments)


def get_clustering_coefficients(ids):
    return list(nx.algorithms.cluster.clustering(graph_only_comments_simple, ids).values())


clustering_coefficients = np.array(make_parallel(get_clustering_coefficients, 
                                                 list(graph_only_comments.nodes()), n_processors))

np.savetxt('clustering_coefficients', clustering_coefficients)
'''
clustering_coefficents

features = pd.DataFrame()
features['user'] = list(graph_only_comments.nodes())
features['in_degrees'] = indegrees_sequence
features['out_degrees'] = outdegrees_sequence
features['out_nbrs_in_degrees_mean'] = out_nbrs_in_degrees_mean
features['out_nbrs_out_degrees_mean'] = out_nbrs_out_degrees_mean
features['in_nbrs_in_degrees_mean'] = in_nbrs_in_degrees_mean
features['in_nbrs_out_degrees_mean'] = in_nbrs_out_degrees_mean
features['nbrs_degrees_mean'] = nbrs_degrees_mean

len(out_nbrs_out_degrees_mean), len(out_nbrs_in_degrees_mean)


data.to_csv('features.csv', index=False)

'''
