import os
import sys
import string
import re

import pickle
import pandas as pd
from pathlib import Path

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

# STEP 1: Corpus Creation

# Declare the filename of the target data to generate the document-term matrix
filepath = 'input/hd_content_FINAL_CyprusMail.csv'
filename = Path(filepath).stem

# Headlines and document contents lists
headlines = []
docs = []

# Read CSV File
data_df = pd.read_csv(filepath)

# Iterate DataFrame to gather the headlines and their content
for index, row in data_df.iterrows():
    if row['Content']:
        headlines.append(row['Headline'])
        docs.append(str(row['Content']).encode(encoding='UTF-8', errors='strict'))

# print(data_df.Content)
# data_df.to_csv('csvs/ex_cleaned_data.csv')
# exit()

# STEP 2: Cleaning Step
round0 = lambda x: clean_lowercase(x)

data_clean = pd.DataFrame(data_df.Content.apply(round0))

round1 = lambda x: clean_code(x)

data_clean = pd.DataFrame(data_clean.Content.apply(round1))

round2 = lambda x: clean_numbers(x)

data_clean = pd.DataFrame(data_clean.Content.apply(round2))

print('Cleaned data: : ' + data_clean)

data_clean.to_csv('csvs/ex_cleaned_data.csv')