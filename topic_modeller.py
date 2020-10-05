# Import the necessary modules for LDA with gensim
# Terminal / Anaconda Navigator: conda install -c conda-forge gensim
import argparse
import os
import sys
from gensim import matutils, models
import scipy.sparse
import pandas as pd
import pickle

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

# Define the following (default) parameters before start topic modeling

# Pickled DataFrame of Document-Term Matrix to Load
pkl_file = 'pickles/dtm_stop_hd_content_FINAL_CyprusMail.pkl'
# pkl_file = 'pickles/dtm_stop_ansa_final_content.pkl'

# Number of topics
topics_num = 10

# Number of iterations
passes_num = 200

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print(
    '___________           .__            _____             .___     .__                \n' +
    '\__    ___/___ ______ |__| ____     /     \   ____   __| _/____ |  |   ___________\n' +
    '  |    | /  _ \\\____ \|  |/ ___\   /  \ /  \ /  _ \ / __ |/ __ \|  | _/ __ \_  __ \\\n' +
    '  |    |(  <_> )  |_> >  \  \___  /    Y    (  <_> ) /_/ \  ___/|  |_\  ___/|  | \/\n' +
    '  |____| \____/|   __/|__|\___  > \____|__  /\____/\____ |\___  >____/\___  >__|   \n' +
    '               |__|           \/          \/            \/    \/          \/       \n')

print("Welcome to RISE TAG Topic Modeller\n")

parser = argparse.ArgumentParser()
parser.add_argument("-file", help="Pickled file to execute LDA analysis and extract topics", type=str)
parser.add_argument("-topics", help="Number of topics to extract", type=int)
parser.add_argument("-passes", help="Number of LDA passes", type=int)
args = parser.parse_args()

if args.file:
    pkl_file = args.file
if args.topics:
    topics_num = args.topics
if args.passes:
    passes_num = args.passes

print('Executing Topic Modeler for ' + str(topics_num) + ' topics in ' + str(
    passes_num) + ' passes on ' + pkl_file + ' ..\n')

# Load pickled DataFrame (Document-Term Matrix)
print('■ Loading pickled DataFrame of Document-Term Matrix..', end='')
data = pd.read_pickle(pkl_file)
print('Done')

# One of the required inputs is a term-document matrix
print('■ Transposing the DataFrame into Term-Document Matrix..', end='')
tdm = data.transpose()
print('Done')
# print(tdm.head())

# We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus
print('■ Converting Term-Document Matrix to gensim format (from df -> sparse matrix -> gensim corpus)..', end='')
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)
print('Done')

# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix
print('■ Loading pickled dictionary of all terms and their respective location in the T-D Matrix..', end='')
cv = pickle.load(open("pickles/cv_stop_hd_content_FINAL_CyprusMail.pkl", "rb"))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())
print('Done')

# Todo: Debug vocabulary here
# print('Vocabulary\n')
# for k in id2word.items():
#     print(k)
# exit()

# Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),
# we need to specify two other parameters as well - the number of topics and the number of passes
print('■ Latent Dirichlet Allocation (LDA) analysis is loading. Please wait..', end='')
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=topics_num, passes=passes_num)
print('Done')
print('\n** ' + str(topics_num) + ' topics found in ' + str(passes_num) + ' passes:\n')
print(lda.print_topics())

# EXTRA STEP: Identify the topic

# # Create a new document-term matrix using only nouns and adjectives, also remove common words with max_df
# add_stop_words = []
# stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
# cvna = CountVectorizer(stop_words=stop_words, max_df=.8)
# data_cvna = cvna.fit_transform(data.Content)
# data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names(), index=data.Headline)
#
# # Let's take a look at which topics each content contains
# corpus_transformed = lda[corpus]
# list(zip([a for [(a,b)] in corpus_transformed], data_dtmna.index))
