# This script generates a document term matrix
import os
import re
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pathlib import Path

# Declare the filename of the target data to generate the document-term matrix
filepath = 'input/hd_content_FINAL_CyprusMail.csv'
filename = Path(filepath).stem

# Headlines and document contents lists
headlines = []
docs = []

# Read CSV File
df = pd.read_csv(filepath)

# Iterate DataFrame to gather the headlines and their content
for index, row in df.iterrows():
    if row['Content']:
        headlines.append(row['Headline'])
        docs.append(str(row['Content']).encode(encoding='UTF-8', errors='strict'))

vec = CountVectorizer()
X = vec.fit_transform(docs)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names(), index=headlines)

# Save corpus vocabulary
# vocabulary = vec.get_feature_names()
# print(vocabulary)
# dictOfWords = dict.fromkeys(vocabulary , 1)
# print(dictOfWords)
# vocab = pd.to_pickle(dictOfWords, 'pickles/vocab_' + filename + '.pkl')
# exit()

# Save pickled DataFrame
df.to_pickle('pickles/'+filename+'.pkl')
# df.to_csv('csvs/dtf_'+filename+'.csv')

# Load pickled DataFrame
df = pd.read_pickle('pickles/'+filename+'.pkl')
print(df)




