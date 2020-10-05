""" Dataset Cleaner

Cleans the given covid news dataset of a source from unwanted text indices, punctuation and
pre-defined text references. During cleaning, it marks the given dataset with semantic tags
focused on covid news numeric references and others.
"""

import argparse
import re
import string

import pickle
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

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

# Intermediate cleaning for specified exclusions
def clean_exclusions(text):
    # remove nan values
    if str(text)=='nan':
        text = re.sub('nan', '', str(text))
    else:
        # The Guardian Pre-specified Exclusions
        text = re.sub(r'^(last modified on |first published on )?(mon|tue|wed|thu|fri|sat|sun) \d{1,2} (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec) \d{4} \d{1,2}\.\d{1,2}\s[a-zA-Z]{3,4}\b(\n|\r|\r\n)', '', text)
        text = re.sub(r'åêåêåê', ' ', text)
        text = re.sub(r'åêåêåê', ' ', text)
        text = re.sub(r'				', ' ', text)
    return text

# Word grouping / replacement
def clean_word_grouping(text):
    # remove nan values
    if str(text)=='nan':
        text = re.sub('nan', '', str(text))
    else:
        # ANSA Pre-specified Tokens for grouping
        text = re.sub('von der leyen', 'von_der_leyen', str(text))

        # NYT Pre-specified Tokens for grouping
        text = re.sub('new york times', 'new_york_times', str(text))
        text = re.sub('new york', 'new_york', str(text))
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
# - [COVAGE] .. Age references
def clean_numbers(text):
    # remove nan values
    if str(text)=='nan':
        text = re.sub('nan', '', str(text))
    else:
        # Covid codename reference
        text = re.sub(r'covid-19', ' [COVCODENAME] ', text)
        text = re.sub(r'covid19', ' [COVCODENAME] ', text)
        text = re.sub(r'SARS-COV(-19)?', ' [COVCODENAME] ', text)
        text = re.sub(r'coronavirus', ' [COVCODENAME] ', text)

        # Currency reference
        text = re.sub(r'(\$|\€|\£){1} ?\d+\.?\d* ?(m|bn|million|billion|trillion)', ' [COVCURRENCY] ', text)
        text = re.sub(r'\d+\.?\d* ?(m|bn|million|billion|trillion)', ' [COVCURRENCY] ', text)
        text = re.sub(r'\d{1,3}((\,|\.)?\d{1,3}?)* ?(m|bn|million|billion|trillion)? ?(\$|\€|\£)', ' [COVCURRENCY] ', text)
        text = re.sub(r'\d{1,3}((\,|\.)?\d{1,3}?)* ?(euro(s)?|dollar(s)?|pound(s)?|eur|EUR|usd|USD|gbp|GBP)', ' [COVCURRENCY] ', text)

        # - [COVDOSE] .. Numbers having mg, milligrams, g, grams
        # - [COVMEASURES] .. Numbers having cm, centimeters, m, meters, k, kilometers

        # Numbers having mg, milligrams, g, grams
        text = re.sub(r'\d{1,3}((\,|\.)?\d{1,3}?)* ?(mg|g|milligrams|grams)\b', ' [COVDOSE] ', text)

        # Numbers having cm, centimeters, m, meters, km, kilometers
        text = re.sub(r'\d{1,3}((\,|\.)?\d{1,3}?)* ?(mm|cm|m|km|millimeter(s)?|millimetre(s)?|centimeter(s)?|centimetre(s)?|meter(s)?|metre(s)?|kilometer(s)?|kilometre(s)?)\b', ' [COVMESAURES] ', text)
        text = re.sub(r'\d+(\,\d+)? ?feet', ' [COVMESAURES] ', text)
        text = re.sub(r'\d{1,2} ?inch(es)?', ' [COVMESAURES] ', text)

        # Time reference
        text = re.sub(r'(\d{1,2}\:\d{2}( ?(a\.?m\.?|p\.?m\.?))?( ?[a-zA-Z]{3,4}((\-|\+)\d{1,2})?)?)|(\d{1,2}( ?(a\.?m\.?|p\.?m\.?)){1}( ?[a-zA-Z]{3,4}((\-|\+)\d{1,2})?)?)', ' [COVTIME] ', text)
        text = re.sub(r'\d+(\,|\.)?(\d+)? ?(year(s)?|day(s)?|month(s)?|week(s)?)', ' [COVTIME] ', text)
        text = re.sub(r'\d+(\-| )?(hour(s)?|minute(s)?|second(s)?|millisecond(s)?)', ' [COVTIME] ', text)
        text = re.sub(r'\d+(\-| )?day', ' [COVTIME] ', text)

        # Date reference
        # TODO: INCLUDE ADDITIONAL PATTERNS (eg. 9/11, 11/2020) --> DONE
        text = re.sub(r'((J|j)anuary|(F|f)ebruary|(M|m)arch|(A|a)pril|(M|m)ay|(J|j)une|(J|j)uly|(A|a)ugust|(S|s)eptember|(O|o)ctober|(N|n)ovember|(D|d)ecember|(J|j)an\.?|(F|f)eb\.?|(M|m)ar\.?|(A|a)pr\.?|(M|m)ay\.?|(J|j)un\.?|(J|j)ul\.?|(A|a)ug\.?|(S|s)ep\.?|(O|o)ct\.?|(N|n)ov\.?|(D|d)ec\.?) \d{1,2}(st|nd|rd|th)?(\,? ((20\d{2})|(19\d{2})))?', ' [COVDATE] ', text)
        text = re.sub(r'((J|j)anuary|(F|f)ebruary|(M|m)arch|(A|a)pril|(M|m)ay|(J|j)une|(J|j)uly|(A|a)ugust|(S|s)eptember|(O|o)ctober|(N|n)ovember|(D|d)ecember){1} \d{1,2}(st|nd|rd|th)?(\, ((20\d{2})|(19\d{2})))?', ' [COVDATE] ', text)
        text = re.sub(r'\d{1,2}(st|nd|rd|th)? ((J|j)anuary|(F|f)ebruary|(M|m)arch|(A|a)pril|(M|m)ay|(J|j)une|(J|j)uly|(A|a)ugust|(S|s)eptember|(O|o)ctober|(N|n)ovember|(D|d)ecember|(J|j)an|(F|f)eb|(M|m)ar|(A|a)pr|(M|m)ay|(J|j)un|(J|j)ul|(A|a)ug|(S|s)ep|(O|o)ct|(N|n)ov|(D|d)ec)( ((20\d{2})|(19\d{2})))?', ' [COVDATE] ', text)
        text = re.sub(r'\d{1,2}(st|nd|rd|th)? ((J|j)anuary|(F|f)ebruary|(M|m)arch|(A|a)pril|(M|m)ay|(J|j)une|(J|j)uly|(A|a)ugust|(S|s)eptember|(O|o)ctober|(N|n)ovember|(D|d)ecember)', ' [COVDATE] ', text)
        text = re.sub(r'\d{1,2}\/(\d{4}|\d{1,2})', ' [COVDATE] ', text)
        text = re.sub(r'\d+(/)+\d+(/)+\d+', ' [COVDATE] ', text)

        # Year reference
        text = re.sub(r'(1900|2000)s', ' [COVYEAR] ', text)
        text = re.sub(r'(20\d{2})|(19\d{2})', ' [COVYEAR] ', text)


        # Day reference
        text = re.sub(r'((M|m)onday|(T|t)uesday|(W|w)ednesday|(T|t)hursday|(F|f)riday|(S|s)aturday|(S|s)unday)', ' [COVDAY] ', text)

        # Month reference
        text = re.sub(r'(J|j)anuary|(F|f)ebruary|(M|m)arch|(A|a)pril|(M|m)ay|(J|j)une|(J|j)uly|(A|a)ugust|(S|s)eptember|(O|o)ctober|(N|n)ovember|(D|d)ecember', ' [COVMONTH] ', text)

        # Age References
        text = re.sub(r'(\d{1,2}\-year\-old)|(\d{1,2} year(s)? old)', ' [COVAGE] ', text)
        text = re.sub(r'mid\-(\d{1}(0s){1})', ' [COVAGE] ', text)
        text = re.sub(r'\d{1}0{1}s{1}', ' [COVAGE] ', text)

        # Other semantics on covid numeric references
        text = re.sub(r'\d+((\,|\.)\d+)? ?(percent|\%)', ' [COVPERCENTAGE] ', text)

        # Coronavirus Deaths references
        text = re.sub(r'\d+( new)?( coronavirus)? death(s)?\b', ' [COVDEATHS] ', text)

        # Coronavirus Tests references
        text = re.sub(r'\d+( new)?( coronavirus| covid)? test(s)?', ' [COVTESTS] ', text)

        # Coronavirus Cases references
        text = re.sub(r'\d+((\,|\.)\d+)*( confirmed)?( coronavirus)? case(s)?', ' [COVCASES] ', text)

        # Coronavirus New Cases references
        text = re.sub(r'\d+((\,|\.)\d+)* new( confirmed)?( coronavirus| covid)?( confirmed)? case(s)?', ' [COVNEWCASES] ', text)

        # Numbers references written in full-text
        text = re.sub(r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirdy|forty|fifty|sixty|seventy|eighty|ninety|hundred(s)?|thousand(s)?|million(s)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\b', ' [COVNUMASWORD] ', text)

        # Number references in parentheses and brackets
        text = re.sub('\[\d+(\.{1,3}|\,)?\d*\]', '[BRANUM]', text)
        text = re.sub('\(\d+(\.{1,3}|\,)?\d*\)', '[PARNUM]', text)

        # Other general numeric references
        text = re.sub('\b\d{1,3}((\,|\.)?\d{1,3}?)*\b', ' [COVNUM] ', text)
        text = re.sub('\b\d+((\.|\,)\d+)*\b', ' [COVNUM] ', text)

        # Other number references fused in words
        text = re.sub('[a-zA-z]+\d+[a-zA-Z]*', '[WORDNUM]', text)
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
    text = re.sub('\n', ' ', text) #TODO: Check the applied fix from nothing to replacing with a space character [!]
    # Convert Trible Spaces into one
    text = re.sub('–', ' ', text)
    # Convert Double Spaces into one
    text = re.sub('  ', ' ', text)
    # Convert Double Spaces into one
    text = re.sub('  ', ' ', text)
    return text


# STEP 1: Corpus Creation

# Declare the filename of the target data to generate the document-term matrix (default value)
filepath = 'input/hd_content_FINAL_CyprusMail.csv'
# filepath = 'input/ansa_final_content.csv'

print(
    '________          __           _________ .__\n' +
    '\______ \ _____ _/  |______    \_   ___ \|  |   ____ _____    ____   ___________\n' +
    ' |    |  \\\__  \\\   __\__  \   /    \  \/|  | _/ __ \\\__  \  /    \_/ __ \_  __ \\\n' +
    ' |    `   \/ __ \|  |  / __ \_ \     \___|  |_\  ___/ / __ \|   |  \  ___/|  | \/\n' +
    '/_______  (____  /__| (____  /  \______  /____/\___  >____  /___|  /\___  >__|\n' +
    '        \/     \/          \/          \/          \/     \/     \/     \/       \n')

print("Welcome to RISE TAG Data Cleaner v.2.0\n")

parser = argparse.ArgumentParser()
parser.add_argument("-file", help="Input CSV file to generate the necessary pickle files for the topic modeling", type=str)
parser.add_argument("-export", help="Output CSV file name of the cleaned dataset", type=str)
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

# TODO: Latest Addition for cleaning specific pre-defined exclusions needed from the datasets
print('■■■ Cleaning pre-defined exclusions..', end='')
round_exclusions = lambda x: clean_exclusions(x)

data_clean = pd.DataFrame(data_clean.Content.apply(round_exclusions))
print('Done')
# TODO: END <<<

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

# TODO: Latest Addition for grouping/replacement of pre-defined word tokens in the datasets
print('■■■ Grouping pre-defined tokens..', end='')
round_grouping = lambda x: clean_word_grouping(x)

data_clean = pd.DataFrame(data_clean.Content.apply(round_grouping))
print('Done')
# TODO: END <<<

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
data_dtm.to_pickle('pickles/corpus_cl_'+filename+'.pkl')
print('[Pickle Saved at pickles/corpus_cl_'+filename+'.pkl]')

if args.export:
    export_path = args.export

    if '.csv' not in export_path:
        export_path += '.csv'

    print('■ Exporting corpus into csv: ' + export_path + '..', end='')
    data_exp = pd.DataFrame()
    data_exp['Headline'] = headlines
    data_exp['Content'] = data_clean.Content
    data_exp.to_csv(export_path)
    print('Done')

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
pickle.dump(cv, open('pickles/cv_stop_'+filename+'.pkl', "wb"))
data_stop.to_pickle('pickles/dtm_stop_'+filename+'.pkl')
print('Done\t\t [Pickles Saved at pickles/cv_stop_'+filename+'.pkl and pickles/dtm_stop_'+filename+'.pkl]')

print('\nData Cleaner Finished. Exiting..')