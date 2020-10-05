# Apply a first round of text cleaning techniques
import argparse
import re
import string

# import pickle
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

# Import the necessary modules for LDA with gensim
# Terminal / Anaconda Navigator: conda install -c conda-forge gensim
import os
import sys
from gensim import matutils, models
import scipy.sparse

# Declare the filename of the target data to generate the document-term matrix (default value)
# filepath = 'input/hd_content_FINAL_CyprusMail.csv'
# filepath = 'input/ansa_final_content.csv'
filepath = 'input/NYT_Content_825.csv'

# Define the following (default) parameters before start topic modeling

# Pickled DataFrame of Document-Term Matrix to Load
# pkl_file = 'pickles/dtm_stop_hd_content_FINAL_CyprusMail.pkl'
# pkl_file = 'pickles/dtm_stop_ansa_final_content.pkl'

# Number of topics
topics_num = 5

# Number of iterations
passes_num = 10

### Data Cleaner

# Remove nan and transform all to lowercase characters
def clean_lowercase(text):
    # remove nan values
    if str(text)=='nan':
        text = re.sub('nan', '', str(text))
    else:
        text = text.lower()
    return text

# Match code patterns and replace them with a [COVCODE] tag
def clean_code(text):
    # remove nan values
    if str(text)=='nan':
        text = re.sub('nan', '', str(text))
    else:
        text = re.sub(r'\!function\(.*\;\b', '[COVCODE]', text)
        text = re.sub(r'if\(.*\)\;', '[COVCODE]', text)
        text = re.sub(r'\w\(.*\)\;', '[COVCODE]', text)
    return text

# Match number patters and replace them with the corresponding tag.
# - [WORDNUM] .. Fused numbers in words (e.g. Covid-19)
# - [BRANUM] .. Numbers enclosed in brackets '[x]'
# - [PARNUM] .. Numbers enclosed in parentheses '(x)'
# - [COVNUM] .. Numbers
# - [COVNUMASWORD] .. Numbers written in full text
# - [COVCODENAME] .. Covid-19 / Covid19 References
# - [COVDAY] .. Fullname Day reference
# - [COVMONTH] .. Fullname Month reference
# - [COVDATE] .. Date reference (eg. 01/06/2020)
# - [COVTIME] .. Time in hours, minutes, seconds and timezones
# - [COVPERCENTAGE] .. Percentages referenced
# - [COVDEATHS] .. Covid Deaths
# - [COVTESTS] .. Covid Tests
# - [COVCASES] .. Covid Cases
# - [COVNEWCASES] .. Covid New Cases
# - [COVCURRENCY] .. Currency amounts references
# - [COVDOSE] .. Numbers having mg, milligrams, g, grams
# - [COVMEASURES] .. Numbers having cm, centimeters, m, meters, km, kilometers
def clean_numbers(text):
    # remove nan values
    if str(text)=='nan':
        text = re.sub('nan', '', str(text))
    else:
        # Covid codename reference
        text = re.sub(r'covid-19', '[COVCODENAME]', text)
        text = re.sub(r'covid19', '[COVCODENAME]', text)

        # Currency reference
        text = re.sub(r'\d+(,)*\d+ euros', ' [COVCURRENCY] ', text)
        text = re.sub(r'\d+(,)*\d+ euro', ' [COVCURRENCY] ', text)
        text = re.sub(r'\d+(,)*\d+ eur', ' [COVCURRENCY] ', text)
        text = re.sub(r'\d+(,)*\d+ dollars', ' [COVCURRENCY] ', text)
        text = re.sub(r'\d+(,)*\d+ dollar', ' [COVCURRENCY] ', text)
        text = re.sub(r'\d+(,)*\d+ usd', ' [COVCURRENCY] ', text)

        text = re.sub(r'[ ^\]]+ euros', ' [COVCURRENCY] ', text)
        text = re.sub(r'[ ^\]]+ euro', ' [COVCURRENCY] ', text)
        text = re.sub(r'[ ^\]]+ eur', ' [COVCURRENCY] ', text)
        text = re.sub(r'[ ^\]]+ dollars', ' [COVCURRENCY] ', text)
        text = re.sub(r'[ ^\]]+ dollar', ' [COVCURRENCY] ', text)
        text = re.sub(r'[ ^\]]+ usd', ' [COVCURRENCY] ', text)

        text = re.sub(r'€[ ^\]]+', ' [COVCURRENCY] ', text)
        text = re.sub(r'\$[ ^\]]+', ' [COVCURRENCY] ', text)
        text = re.sub(r'\d+(,)*\d+ €', ' [COVCURRENCY] ', text)
        text = re.sub(r'\d+(,)*\d+€', ' [COVCURRENCY] ', text)
        text = re.sub(r'€\d+(,)*\d+', ' [COVCURRENCY] ', text)
        text = re.sub(r'€ \d+(,)*\d+', ' [COVCURRENCY] ', text)
        text = re.sub(r'\d+(,)*\d+ \$', ' [COVCURRENCY] ', text)
        text = re.sub(r'\d+(,)*\d+\$', ' [COVCURRENCY] ', text)
        text = re.sub(r'\$\d+(,)*\d+', ' [COVCURRENCY] ', text)
        text = re.sub(r'\$ \d+(,)*\d+', ' [COVCURRENCY] ', text)

        # - [COVDOSE] .. Numbers having mg, milligrams, g, grams
        # - [COVMEASURES] .. Numbers having cm, centimeters, m, meters, k, kilometers

        # Numbers having mg, milligrams, g, grams
        text = re.sub(r'\d+ mg\b', ' [COVDOSE] ', text)
        text = re.sub(r'\d+mg\b', ' [COVDOSE] ', text)
        text = re.sub(r'\d+ milligrams\b', ' [COVDOSE] ', text)
        text = re.sub(r'\d+ g\b', ' [COVDOSE] ', text)
        text = re.sub(r'\d+g\b', ' [COVDOSE] ', text)
        text = re.sub(r'\d+ grams\b', ' [COVDOSE] ', text)

        # Numbers having cm, centimeters, m, meters, km, kilometers
        text = re.sub(r'\d+ cm\b', ' [COVMESAURES] ', text)
        text = re.sub(r'\d+cm\b', ' [COVMESAURES] ', text)
        text = re.sub(r'\d+ centimeters\b', ' [COVMESAURES] ', text)
        text = re.sub(r'\d+ m\b', ' [COVMESAURES] ', text)
        text = re.sub(r'\d+m\b', ' [COVMESAURES] ', text)
        text = re.sub(r'\d+ meters\b', ' [COVMESAURES] ', text)
        text = re.sub(r'\d+ km\b', ' [COVMESAURES] ', text)
        text = re.sub(r'\d+km\b', ' [COVMESAURES] ', text)
        text = re.sub(r'\d+ kilometers\b', ' [COVMESAURES] ', text)

        # Time reference
        text = re.sub(r'\d+(,)*\d+ hours', ' [COVTIME] ', text)
        text = re.sub(r'\d+(,)*\d+ hour', ' [COVTIME] ', text)
        text = re.sub(r'\d+(,)*\d+ minutes', ' [COVTIME] ', text)
        text = re.sub(r'\d+(,)*\d+ minute', ' [COVTIME] ', text)
        text = re.sub(r'\d+(,)*\d+ seconds', ' [COVTIME] ', text)
        text = re.sub(r'\d+(,)*\d+ second', ' [COVTIME] ', text)
        text = re.sub(r'\d+(,)*\d+ days', ' [COVTIME] ', text)
        text = re.sub(r'\d+(,)*\d+ day', ' [COVTIME] ', text)
        text = re.sub(r'\d+(,)*\d+ months', ' [COVTIME] ', text)
        text = re.sub(r'\d+(,)*\d+ month', ' [COVTIME] ', text)
        text = re.sub(r'\d+(,)*\d+ years', ' [COVTIME] ', text)
        text = re.sub(r'\d+(,)*\d+ year', ' [COVTIME] ', text)

        # Timezone reference aggregated as time
        text = re.sub(r'\d+(:)*\d+ gmt', ' [COVTIME] ', text)
        text = re.sub(r'\d+(:)*\d+ utc', ' [COVTIME] ', text)
        text = re.sub(r'\d+(:)*\d+ eest', ' [COVTIME] ', text)
        text = re.sub(r'\d+(:)*\d+ est', ' [COVTIME] ', text)
        text = re.sub(r'\d+(:)*\d+ cst', ' [COVTIME] ', text)
        text = re.sub(r'\d+(:)*\d+ ast', ' [COVTIME] ', text)
        text = re.sub(r'\d+(:)*\d+ mst', ' [COVTIME] ', text)
        text = re.sub(r'\d+(:)*\d+ pst', ' [COVTIME] ', text)
        text = re.sub(r'\d+(:)*\d+ akst', ' [COVTIME] ', text)
        text = re.sub(r'\d+(:)*\d+ hst', ' [COVTIME] ', text)

        # Date reference
        text = re.sub(r'\d+(/)+\d+(/)+\d+', ' [COVDATE] ', text)

        # Year reference
        text = re.sub(r'2019', ' [COVYEAR] ', text)
        text = re.sub(r'2020', ' [COVYEAR] ', text)
        text = re.sub(r'2021', ' [COVYEAR] ', text)

        # Day reference
        text = re.sub(r'monday', ' [COVDAY] ', text)
        text = re.sub(r'tuesday', ' [COVDAY] ', text)
        text = re.sub(r'wednesday', ' [COVDAY] ', text)
        text = re.sub(r'thursday', ' [COVDAY] ', text)
        text = re.sub(r'friday', ' [COVDAY] ', text)
        text = re.sub(r'saturday', ' [COVDAY] ', text)
        text = re.sub(r'sunday', ' [COVDAY] ', text)

        # Month reference
        text = re.sub(r'january', ' [COVMONTH] ', text)
        text = re.sub(r'february', ' [COVMONTH] ', text)
        text = re.sub(r'march', ' [COVMONTH] ', text)
        text = re.sub(r'april', ' [COVMONTH] ', text)
        text = re.sub(r'may', ' [COVMONTH] ', text)
        text = re.sub(r'june', ' [COVMONTH] ', text)
        text = re.sub(r'july', ' [COVMONTH] ', text)
        text = re.sub(r'august', ' [COVMONTH] ', text)
        text = re.sub(r'september', ' [COVMONTH] ', text)
        text = re.sub(r'october', ' [COVMONTH] ', text)
        text = re.sub(r'november', ' [COVMONTH] ', text)
        text = re.sub(r'december', ' [COVMONTH] ', text)

        # Other semantics on covid numeric references
        text = re.sub(r'\d+(,)*\d+%', ' [COVPERCENTAGE] ', text)
        text = re.sub(r'\d+(.)*\d+%', ' [COVPERCENTAGE] ', text)
        text = re.sub(r'\d+(,)*\d+ percent', ' [COVPERCENTAGE] ', text)
        text = re.sub(r'\d+(.)*\d+ percent', ' [COVPERCENTAGE] ', text)
        text = re.sub(r'\d+,\d+ deaths', ' [COVDEATHS] ', text)
        text = re.sub(r'\d+ deaths', ' [COVDEATHS] ', text)
        text = re.sub(r'\d+,\d+ tests', ' [COVTESTS] ', text)
        text = re.sub(r'\d+ tests', ' [COVTESTS] ', text)
        text = re.sub(r'\d+,\d+ cases', ' [COVCASES] ', text)
        text = re.sub(r'\d+ cases', ' [COVCASES] ', text)
        text = re.sub(r'\d+ new cases', ' [COVNEWCASES] ', text)
        text = re.sub(r'\d+ new case', ' [COVNEWCASES] ', text)

        text = re.sub(r'\bone\b(\.)?', ' [COVNUMASWORD] ', text)
        text = re.sub(r'\btwo\b(\.)?', ' [COVNUMASWORD] ', text)
        text = re.sub(r'\bthree\b(\.)?', ' [COVNUMASWORD] ', text)
        text = re.sub(r'\bfour\b(\.)?', ' [COVNUMASWORD] ', text)
        text = re.sub(r'\bfive\b(\.)?', ' [COVNUMASWORD] ', text)
        text = re.sub(r'\bsix\b(\.)?', ' [COVNUMASWORD] ', text)
        text = re.sub(r'\bseven\b(\.)?', ' [COVNUMASWORD] ', text)
        text = re.sub(r'\beight\b(\.)?', ' [COVNUMASWORD] ', text)
        text = re.sub(r'\bnine\b(\.)?', ' [COVNUMASWORD] ', text)
        text = re.sub(r'\bten\b(\.)?', ' [COVNUMASWORD] ', text)

        text = re.sub(r'eleven', ' [COVNUMASWORD] ', text)
        text = re.sub(r'twelve', ' [COVNUMASWORD] ', text)
        text = re.sub(r'thirteen', ' [COVNUMASWORD] ', text)
        text = re.sub(r'fourteen', ' [COVNUMASWORD] ', text)
        text = re.sub(r'fifteen', ' [COVNUMASWORD] ', text)
        text = re.sub(r'sixteen', ' [COVNUMASWORD] ', text)
        text = re.sub(r'sixteen', ' [COVNUMASWORD] ', text)
        text = re.sub(r'eighteen', ' [COVNUMASWORD] ', text)
        text = re.sub(r'nineteen', ' [COVNUMASWORD] ', text)

        text = re.sub(r'twenty', ' [COVNUMASWORD] ', text)
        text = re.sub(r'thirty', ' [COVNUMASWORD] ', text)
        text = re.sub(r'forty', ' [COVNUMASWORD] ', text)
        text = re.sub(r'fifty', ' [COVNUMASWORD] ', text)
        text = re.sub(r'sixty', ' [COVNUMASWORD] ', text)
        text = re.sub(r'seventy', ' [COVNUMASWORD] ', text)
        text = re.sub(r'eighty', ' [COVNUMASWORD] ', text)
        text = re.sub(r'ninety', ' [COVNUMASWORD] ', text)

        text = re.sub(r'hundreds', ' [COVNUMASWORD] ', text)
        text = re.sub(r'thousands', ' [COVNUMASWORD] ', text)
        text = re.sub(r'millions', ' [COVNUMASWORD] ', text)
        text = re.sub(r'hundred', ' [COVNUMASWORD] ', text)
        text = re.sub(r'thousand', ' [COVNUMASWORD] ', text)
        text = re.sub(r'million', ' [COVNUMASWORD] ', text)

        text = re.sub(r'first', ' [COVNUMASWORD] ', text)
        text = re.sub(r'second', ' [COVNUMASWORD] ', text)
        text = re.sub(r'third', ' [COVNUMASWORD] ', text)
        text = re.sub(r'fourth', ' [COVNUMASWORD] ', text)
        text = re.sub(r'fifth', ' [COVNUMASWORD] ', text)
        text = re.sub(r'sixth', ' [COVNUMASWORD] ', text)
        text = re.sub(r'seventh', ' [COVNUMASWORD] ', text)
        text = re.sub(r'eighth', ' [COVNUMASWORD] ', text)
        text = re.sub(r'ninth', ' [COVNUMASWORD] ', text)
        text = re.sub(r'tenth', ' [COVNUMASWORD] ', text)
        text = re.sub(r'eleventh', ' [COVNUMASWORD] ', text)
        text = re.sub(r'twelfth', ' [COVNUMASWORD] ', text)

        text = re.sub('\d+,\d+', ' [COVNUM] ', text)
        text = re.sub('\d+(\.)*\d+', ' [COVNUM] ', text)
        text = re.sub('\d+(/)*\d+', ' [COVNUM] ', text)

        # Number references in parentheses and brackets
        text = re.sub('\[\d*?\]', '[BRANUM]', text)
        text = re.sub('\(\d*?\)', '[PARNUM]', text)

        # Other number references fused in words
        text = re.sub('\w*\d\w*', '[WORDNUM]', text)
        # text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    # remove nan values
    if str(text)=='nan':
        text = re.sub('nan', '', str(text))
    else:
        text = re.sub('\w*\d\w*', '', text)
        # text = text.lower()
        # text = re.sub('\[.*?\]', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text


# Apply a second round of cleaning
def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text


# STEP 1: Corpus Creation

print(
' ____ ___      .__  _____.__           .___ ___________           .__            _____             .___     .__  .__                \n' +
'|    |   \____ |__|/ ____\__| ____   __| _/ \__    ___/___ ______ |__| ____     /     \   ____   __| _/____ |  | |  |   ___________ \n' +
'|    |   /    \|  \   __\|  |/ __ \ / __ |    |    | /  _ \\____ \|  |/ ___\   /  \ /  \ /  _ \ / __ |/ __ \|  | |  | _/ __ \_  __ \\\n' +
'|    |  /   |  \  ||  |  |  \  ___// /_/ |    |    |(  <_> )  |_> >  \  \___  /    Y    (  <_> ) /_/ \  ___/|  |_|  |_\  ___/|  | \/\n' +
'|______/|___|  /__||__|  |__|\___  >____ |    |____| \____/|   __/|__|\___  > \____|__  /\____/\____ |\___  >____/____/\___  >__|\n' +
'             \/                  \/     \/                 |__|           \/          \/            \/    \/               \/       \n')

print("Welcome to RISE TAG Unified Topic Modeller\n")

parser = argparse.ArgumentParser()
parser.add_argument("-file", help="Pickled file to execute LDA analysis and extract topics", type=str)
parser.add_argument("-topics", help="Number of topics to extract", type=int)
parser.add_argument("-passes", help="Number of LDA passes", type=int)
args = parser.parse_args()

if args.file:
    filepath = args.file

# Take only the name of the given file
filename = Path(filepath).stem

# Headlines and document contents lists
headlines = []
docs = []

# Read CSV File
print('■ Reading CSV File (' + filepath + ') into DataFrame..', end='')
data_df = pd.read_csv(filepath)
print('Done')

# Iterate DataFrame to gather the headlines and their content
print('■ Parsing headlines and contents..', end='')
for index, row in data_df.iterrows():
    if row['Content']:
        headlines.append(row['Headline'])
        docs.append(str(row['Content']).encode(encoding='UTF-8', errors='strict'))
print('Done')

print('■ Removing English Stop-Words..', end='')
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(docs)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names(), index=headlines)
print('Done')
# print(data_dtm)

# STEP 2: Cleaning Step
print('■ Cleaning data contents..')

# -->>> Additional cleaning steps for numeric references and code snippets cleaning

print('■■■ Lowercase transformation..', end='')
round_lw = lambda x: clean_lowercase(x)

data_clean = pd.DataFrame(data_df.Content.apply(round_lw))
print('Done')

print('■■■ Marking code snippets..', end='')
round_code = lambda x: clean_code(x)

data_clean = pd.DataFrame(data_clean.Content.apply(round_code))
print('Done')

print('■■■ Marking numeric references..', end='')
round_nums = lambda x: clean_numbers(x)

data_clean = pd.DataFrame(data_clean.Content.apply(round_nums))
print('Done')

# --<<<

print('■■■ Cleaning refinements [1st Pass]..', end='')
round1 = lambda x: clean_text_round1(x)

data_clean = pd.DataFrame(data_clean.Content.apply(round1))
print('Done')
# print('Cleaning Step Pass #1: ' + data_clean)

print('■■■ Cleaning refinements [2nd Pass]..', end='')
round2 = lambda x: clean_text_round2(x)

data_clean = pd.DataFrame(data_clean.Content.apply(round2))
print('Done')
# print('Cleaning Step Pass #2: ' + data_clean)

# STEP 3: Generate Document-Term Matrix
# Create a document-term matrix using CountVectorizer, and exclude common English stop words
print('■ Generating Document-Term Matrix using sklearn..', end='')
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.Content)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names(),index=headlines)
print(data_dtm)
print('Done\t\t', end='')

# Save cleaned corpus
# Let's pickle it for later use
# data_dtm.to_pickle('pickles/corpus_cl_'+filename+'.pkl')
# print('[Pickle Saved at pickles/corpus_cl_'+filename+'.pkl]')

# STEP 4: Generate Vocabulary

print('■ Generating Vocabulary using sklearn..', end='')
# Read in cleaned data
# data_clean = pd.read_pickle('data_clean.pkl')

# If more than half of the media have it as a top word, exclude it from the list (optional step for later)
add_stop_words = ['said', 'åêåêåê']

# Add new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Todo: Debug stop words here
# print('Stop words:\n')
# print(cv.get_feature_names())

# Recreate document-term matrix
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.Content)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names(), index=headlines)

# Pickle it for later use
# pickle.dump(cv, open('pickles/cv_stop_'+filename+'.pkl', "wb"))
# data_stop.to_pickle('pickles/dtm_stop_'+filename+'.pkl')
# print('Done\t\t [Pickles Saved at pickles/cv_stop_'+filename+'.pkl and pickles/dtm_stop_'+filename+'.pkl]')

print('\nData Cleaner Finished.')





### Topic Modeller

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("Topic Modelling Process Started.\n")

if args.file:
    pkl_file = args.file
if args.topics:
    topics_num = args.topics
if args.passes:
    passes_num = args.passes

print('Executing Topic Modeler for ' + str(topics_num) + ' topics in ' + str(
    passes_num) + ' passes on ' + filepath + ' ..\n')

# Load pickled DataFrame (Document-Term Matrix)
print('■ Loading DataFrame of Document-Term Matrix..', end='')
print('Done')

# One of the required inputs is a term-document matrix
print('■ Transposing the DataFrame into Term-Document Matrix..', end='')
tdm = data_stop.transpose()
print('Done')
# print(tdm.head())

# We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus
print('■ Converting Term-Document Matrix to gensim format (from df -> sparse matrix -> gensim corpus)..', end='')
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)
print('Done')

# Gensim also requires dictionary of the all terms and their respective location in the term-document matrix
print('■ Loading dictionary of all terms and their respective location in the T-D Matrix..', end='')
# cv = pickle.load(open("pickles/cv_stop_hd_content_FINAL_CyprusMail.pkl", "rb"))
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
