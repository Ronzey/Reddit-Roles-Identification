-------------------------
   User Representation
-------------------------
Each user are identified by a unique user id, followed by a vector
to represent its features. Here we take four categories of features
to represent each redditor.

1. LIWC
liwc_pre.py  
It only process the comments body, so it is totally dataset
compatible. Note that the google bigquery imported data type is
.csv or .json, we need to transform the data into .txt, which the
LIWC application program can handle. Once we get the features analyzed
by LIWC, it is stored in .txt files, so we need to integrate these
features in corresponding users.


2. Topics
topic_script2.ipynb
    Topics are also based on comments body, which means this feature
are dataset compatible. The topic is a more specific division than
the subreddits, which is an attribute of a comment. The LDA model are
used for this sub-task. In a specific entry of the user representation
vector, we can assign each topic a integer.


3. Network Analysis
Required input
The Reddit comments collected from
https://bigquery.cloud.google.com/dataset/fh-bigquery:reddit_comments

Scripts
generate_features.py: script for generating first-order features. Features will be outputed to multiple files, e.g. indegree, inmulti, etc. A separate file, nodes, will include all the nodes in the order followed by all other feature files. 

generate_nbr_features.py: script for generating neighborhood features. Features will be outputed to multiple files as well. 

combine.py: take all generated feature files and node file, and combine them into one .csv file, and save to features.csv.

4. Discourse Acts
    It is based on the MIT discourse data set. We train a model on
that data set, and then we expect it having good generalization ability
to process the data scraped from the Google BigQuery tables.

Dataset Download Link: https://github.com/google-research-datasets/coarse-discourse/blob/master/coarse_discourse_dataset.json

Instruction: followed the instruction in the .ipynb file to run the code

5. GMM
gmm.py - to calculate the AIC and BIC and generate roles clustering


