
nodes = 'nodes'
features = ['indegree', 'outdegree', 'inmulti', 'outmulti', 'in_nbrs_out_degrees_mean',
        'in_nbrs_in_degrees_mean', 'out_nbrs_out_degrees_mean', 'out_nbrs_in_degrees_mean',
        'clustering_coefficients']


import pandas as pd

data = pd.DataFrame()
with open(nodes, 'r') as f:
    l = [x.strip() for x in f.readlines()]
    data[nodes] = l

for feat in features:
    with open(feat, 'r') as f:
        l = [float(x) for x in f.readlines()]
        data[feat] = l

data.to_csv('features.csv', index=None)
