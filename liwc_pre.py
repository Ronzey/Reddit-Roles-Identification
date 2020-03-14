# Find the authors who are in the author_rep list,
# and look for their comments in the corresponding month,
# extend the representation with the liwc analysis results

import praw
from praw import models
import pandas
from util import in_duration
import pprint



reddit = praw.Reddit(client_id='O3eCkJp5u4-S-Q',
                     client_secret='zBLTi3qGzBn0qdkvvA4Fdf0NZGE',
                     password='777zrz777',
                     user_agent='RonzeyZhang v1.0 by /u/RonzeyZhang',
                     username='RonzeyZhang')

month = '2019_01'
root_path = 'D:/Research/roleIdentification/dataset'
authors_path = '{}/author_{}_rep'.format(root_path,month)
comments_path = '{}/comment_{}.csv'.format(root_path,month)

authors_file = open(authors_path, encoding='utf8')
comments_file = open(comments_path, encoding='utf8')

count = 0
comments_csv = pandas.read_csv(comments_path)

for line in authors_file.readlines()[1:]:
    author_name = line.split(',')[0] # prevent the same sub strings are found in comment_body field
    title = author_name
    text_output = ''
    redditor = praw.models.Redditor(reddit, name = author_name)
    author_comments = []
    redditor_comments = comments_csv['body'][comments_csv['author'] == author_name]
    # count the number of each user's comments in a specific time period
    title += ' ' + str(redditor_comments.values.size) # number of comments
    for comment in redditor_comments.values:
        comment = ' '.join(comment.split())
        text_output += (comment + '\n')
    count += 1
    print(count)
    author_comments_path = '{}/authors_comments_{}/{}.txt'.format(root_path, month, title)
    author_comments_file = open(author_comments_path, 'w', encoding='utf8')
    author_comments_file.write(text_output)
    author_comments_file.close()





