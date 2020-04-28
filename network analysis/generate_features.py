
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

n_processors = 50
import os

'''
files = [os.path.join('2019_02', x) for x in os.listdir('2019_02')]

data_chunks = []
for f in files[:1]:
    data_chunks.append(pd.read_csv(f))

print('reading is done.')

data = pd.concat(data_chunks)

print('n comments', len(data))
print('n authors', len(set(data.author)))

is_politics = data.subreddit == 'politics'

columns_preserved = ['id', 'parent_id', 'author', 'created_utc']
data = data[columns_preserved]

parent_types, parent_ids = zip(*[t.split('_') for t in data.parent_id])
data['parent_types'] = parent_types
data['parent'] = parent_ids

og_ids = data['id']
data = data[data.parent_types=='t1']
'''

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

'''
comment_author_table = make_parallel(build_comment_author_table, list(range(len(data))), n_processors)

comment_author_table = dict(comment_author_table)

def to_edgelists(data_to_use, verbose=False):
    edges = []
    edges_only_comments = []
    edges_to_links = []
    start_time = time.time()
    if verbose: 
        print('start collecting edges')
    for i, entry in data_to_use.iterrows():
        parent_id = entry.parent_id
        ptype, parent = parent_id.split('_')
        try:
            parent = comment_author_table[parent]
        except KeyError:
            continue
        src = comment_author_table[entry.id]
        #edges.append([entry.id, parent, {'type': ptype}])

        if ptype == 't1':
            edges_only_comments.append([src, parent])
        elif ptype == 't3':
            #edges_to_links.append([entry.id, parent])
            pass
        else:
            if verbose:
                print('unknown ptype: ' + ptype)

        if verbose:
            elapsed = time.time() - start_time
            if i % 10 == 9:  # in case output rate exceeds notebook limit
                sys.stdout.write('\r{}/{} {:.2f}% processed, time elapsed: {}, eta: {}'.format(
                    i + 1, len(data_to_use), 100.0 * (i + 1) / len(data_to_use), str(
                        datetime.timedelta(seconds=np.round(elapsed, 2))),
                    str(
                        datetime.timedelta(seconds=np.round( elapsed * len(data_to_use) / (i+1) - elapsed , 2)
                        ))
                    )
                )
                sys.stdout.flush()
    #return edges, edges_only_comments, edges_to_links
    return edges_only_comments

len_split = int(len(data)/n_processors)
indices = [[i*len_split, (i+1)*len_split] for i in range(n_processors)]
indices[-1][-1] = len(data)

data_split = [data.iloc[x[0]:x[1], :] for x in indices]


start = time.time()
edges = []
num_tasks = len(data_split)
with Pool(n_processors) as p:
    for i, x in enumerate(p.imap_unordered(to_edgelists, data_split, 1)):
        edges += (x)
        sys.stderr.write('\rdone {:%}, remaining: {}'.format(
            (i+1)/num_tasks, num_tasks - (i+1)
        ))
        sys.stderr.flush()
    #res = p.imap_unordered(to_edgelists, data_split)
print('')
print(time.time() - start)

with open('edgelist', 'w') as f:
    for e in edges:
        f.write('{} {}'.format(e[0], e[1]))
        f.write('\n')

#start = time.time()
#res = to_edgelists(data)
#print(time.time() - start)

'''
with open('edgelist_large', 'r') as f:
    edges = [x.split() for x in f.readlines()]
graph_only_comments = nx.MultiDiGraph()


edge_keys = graph_only_comments.add_edges_from(edges)
# ### First, let's look at degrees and degree distribution.
# 
# In this step, let's consider only the source nodes of data entrys (data.id). Let's see their in degree and out degree distribution.

# In[ ]:
with open('nodes', 'w') as f:
    f.write('\n'.join(list(graph_only_comments.nodes())))


def get_in_degree(x):
    return list(dict(graph_only_comments.in_degree(x)).values())
indegrees_sequence = make_parallel(
        get_in_degree, list(graph_only_comments.nodes()), n_processors)

indegrees_sequence = np.array(indegrees_sequence)


# In[ ]:


from collections import defaultdict
def hist_log_log(x):
    freq = defaultdict(lambda : 0)
    for i in x:
        freq[i] += 1
    vals = np.array(list(freq.keys()))
    freqs = np.array(list(freq.values()))
    
    plt.scatter(np.log(1+vals), np.log(1+freqs), c='k', s=3)
    
    plt.show()
        


# In[ ]:


#hist_log_log(indegrees_sequence)


# In[ ]:


print(np.mean(list(indegrees_sequence)), np.max(indegrees_sequence), np.std(indegrees_sequence))
print('{:.3f}%, {} nodes have in-degree more than 1.'.format(
    100.0 * np.sum(indegrees_sequence > 1) / len(indegrees_sequence),
    np.sum(indegrees_sequence > 1)))

np.savetxt('indegree', indegrees_sequence)


# In[ ]:


def get_out_degree(x):
    return list(dict(graph_only_comments.out_degree(x)).values())


# In[ ]:

outdegrees_sequence = np.array(make_parallel(
        get_out_degree, list(graph_only_comments.nodes()), n_processors))
np.savetxt('outdegree', outdegrees_sequence)


print(np.mean(list(outdegrees_sequence)), np.max(outdegrees_sequence), np.std(outdegrees_sequence))
print('{:.3f}%, {} nodes have out-degree more than 1.'.format(
    100.0 * np.sum(outdegrees_sequence > 1) / len(outdegrees_sequence),
    np.sum(outdegrees_sequence > 1)))


# In[ ]:

def generate_in_multiplicity(idx):

    in_multiplicity = [
         len(graph_only_comments.in_edges(x)) / len(set([t[0] for t in graph_only_comments.in_edges(x)]))
        if len(set([t[0] for t in graph_only_comments.in_edges(x)])) else 0
        for x in idx
    ]

    return in_multiplicity

def generate_out_multiplicity(idx):

    out_multiplicity = [
         len(graph_only_comments.out_edges(x)) / len(set([t[1] for t in graph_only_comments.out_edges(x)]))
        if len(set([t[1] for t in graph_only_comments.out_edges(x)])) else 0
        for x in idx
    ]

    return out_multiplicity


in_multiplicity = make_parallel(generate_in_multiplicity, list(graph_only_comments.nodes()),
        n_processors)
print('in mul:')
print(np.mean(in_multiplicity), np.min(in_multiplicity), np.max(in_multiplicity),
        np.std(in_multiplicity))
np.savetxt('inmulti', in_multiplicity)
out_multiplicity = make_parallel(generate_out_multiplicity, list(graph_only_comments.nodes()),
        n_processors)
np.savetxt('outmulti', out_multiplicity)

# In[ ]:
