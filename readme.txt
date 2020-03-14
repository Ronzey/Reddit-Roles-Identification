---------------
    Dataset
---------------
We consider to scrape comments through python scripts for a coarse
trial, where praw can provide us some powerful interface.
However, the data source finally we use should be queried from the
Google BigQuery.

Preprocessing is necessary, the method used by Amy et.al. can be
a reference.

-------------------------
   User Representation
-------------------------
Each user are identified by a unique user id, followed by a vector
to represent its features. Here we take four categories of features
to represent each redditor.

1. LIWC
    It only process the comments body, so it is totally dataset
compatible. Note that the google bigquery imported data type is
.csv or .json, we need to transform the data into .txt, which the
LIWC application program can handle. Once we get the features analyzed
by LIWC, it is stored in .txt files, so we need to integrate these
features in corresponding users.

2. Topics
    Topics are also based on comments body, which means this feature
are dataset compatible. The topic is a more specific division than
the subreddits, which is an attribute of a comment. The LDA model are
used for this sub-task. In a specific entry of the user representation
vector, we can assign each topic a integer.
Question: Do you need to regularize the integer? Is it enough to use
just one dimension.

3. Interaction
    Something about the graphic model. Have not take a deeper look.

4. Discourse Acts
    It is based on the MIT discourse data set. We train a model on
that data set, and then we expect it having good generalization ability
to process the data scraped from the Google BigQuery tables.

-------------------------
          To Do
-------------------------
Now let's playing around with the fuxking Google BigQuery!
