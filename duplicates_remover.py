import os
import sys

import pandas as pd
import textdistance

# TODO: Note that it seems that the similarity of 0.80 and above refers to the same article

filepath = 'input/ansa_final_content.csv'
similarity_threshold = 0.80
export_path = 'output/ansa_final_content_dup_cn.csv'

# Read CSV File
print('■ Reading CSV File (' + filepath + ') into DataFrame..', end='')
data_df = pd.read_csv(filepath, low_memory=False)
print('Done')

index = data_df.index
number_of_rows = len(index)
# print('Number of records loaded: ' + str(number_of_rows))

# Iterate DataFrame to gather the headlines and their content
print('■ Calculating headline similarities..', end='')

droped_records = 0
previous_headline = None
for index, row in data_df.iterrows():
    if row['Content']:
        if previous_headline is None:
            previous_headline = str(row['Headline'])
        else:
            jaccard_score = textdistance.jaccard(previous_headline, str(row['Headline']))
            # Discard record if the Jaccard similarity score is above or equal the threshold (of 0.80 - recommended)
            if jaccard_score >= similarity_threshold:
                data_df.drop(index, inplace=True)
                droped_records += 1
            else:
                # If the headlines are identified as different, assign a new previous headline
                previous_headline = str(row['Headline'])
print('Done')

print('Similar Records Dropped: ' + str(droped_records) + ' out of ' + str(number_of_rows) + ' loaded')
print('Remaining Records in dataset: ' + str(len(data_df.index)))

if '.csv' not in export_path:
    export_path += '.csv'
print('■ Exporting processed CSV file: ' + export_path + '.. ', end='')
data_df.to_csv(export_path)
print('Done')

print('\nDuplicates Remover finished processing. Exiting..')
exit()
tokens1 = 'Coronavirus: Toughest days of our lives says Speranza'
tokens2 = 'Coronavirus: Toughest days of our lives says Speranza (7)'

tokens1 = '++ Coronavirus: deaths up by 47 ++ (3)'
tokens2 = '++ Coronavirus: deaths up by 47 ++'

tokens1 = 'Coronavirus: 251 new cases in Italy (5)'
tokens2 = '++ Coronavirus: 251 new cases in Italy ++'

tokens1 = 'Coronavirus: intensive-care cases up after weeks (2)'
tokens2 = '++ Coronavirus: intensive-care cases up ++'

tokens1 = 'Coronavirus: 333 new cases in Italy (2)'
tokens2 = 'Coronavirus: 329 new cases in Italy (2)'

tokens1 = 'Coronavirus: No signs of 2nd wave-Sileri'
tokens2 = 'Coronavirus: No signs of 2nd wave of contagion says Sileri (4)'

score = textdistance.jaccard(tokens1, tokens2)

print(str(score))

