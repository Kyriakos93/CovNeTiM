# Apply a first round of text cleaning techniques
import argparse
import re
import string

# import pickle
import pandas as pd
from pathlib import Path

from gensim.models import CoherenceModel
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

# Import the necessary modules for LDA with gensim
# Terminal / Anaconda Navigator: conda install -c conda-forge gensim
import os
import sys
from gensim import matutils, models, corpora
import scipy.sparse

import swifter
import matplotlib.pyplot as plt

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

# Intermediate cleaning for specified exclusions
def clean_exclusions(text):
    # remove nan values
    if str(text)=='nan':
        text = re.sub('nan', '', str(text))
    else:
        # Global Exclusions
        text = re.sub(r'said', ' ', text)
        # The Guardian Pre-defined Exclusions
        # text = re.sub(r'^(last modified on |first published on )?(mon|tue|wed|thu|fri|sat|sun) \d{1,2} (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec) \d{4} \d{1,2}\.\d{1,2}\s[a-zA-Z]{3,4}\b(\n|\r|\r\n)', '', text)
        text = re.sub(r'(last modified on |first published on )?(mon|tue|wed|thu|fri|sat|sun) \d{1,2} (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec) \d{4} \d{1,2}\.\d{1,2}\s[a-zA-Z]{3,4}', '', text)
        # ANSA Pre-defined Exclusions
        text = re.sub(r'\(ansa\) \- \w[A-Za-z]+\, (\d{1,2} \w[A-Za-z]+|\w[A-Za-z]+ \d{1,2})', ' ', text)
        text = re.sub(r'ansa', ' ', text)
        text = re.sub(r'¬†', ' ', text)
        text = re.sub(r'Â', ' ', text)
        text = re.sub(r'åêåêåê', ' ', text)
        text = re.sub(r'åêåêåê', ' ', text)
        text = re.sub(r'				', ' ', text)
        # NYT Pre-defined Exclusions
        text = re.sub(r'supported by', '', text)

        # Remove unwanted tokens causing Catastrophic Backtracking for Regular Expressions
        text = re.sub(r'(\+\d+)+', ' ', text)

        # Remove Huge Decimals (e.g. full-written pi) or nunbers that cause Catastrophic Backtracking
        # for Regural Expressions
        text = re.sub(r'\d{1,3}((\,|\.)(\d{11})+)', ' ', text)
        text = re.sub(r'\d{38}', ' ', text)
        text = re.sub(r'\d{36}', ' ', text)
        text = re.sub(r'\d{34}', ' ', text)
        text = re.sub(r'\d{1,3}((\,|\.)\d{3,7}){9,11}', ' ', text)
        text = re.sub(r'\d{20}((\,|\.)\d{3,4})+', ' ', text)
        text = re.sub(r'\%\d{20}(\d{1,5})*', ' ', text)
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
        # Debug input string
        # print('NEW_TEXT>>>' + text)

        # Currency reference
        # text = re.sub(r'\d+\.?\d* ?(m|bn|million|billion|trillion)', ' [COVCURRENCY] ', text)
        text = re.sub(r'\d{1,3}((\,|\.)?\d{1,3}?)* ?(m|bn|million|billion|trillion)? ?(\$|\€|\£)', ' [COVCURRENCY] ', text)
        text = re.sub(r'\d{1,3}((\,|\.)?\d{1,3}?)* ?(m|bn|million|billion|trillion)? ?(euro(s)?|dollar(s)?|pound(s)?|eur|EUR|usd|USD|gbp|GBP)', ' [COVCURRENCY] ', text)
        # text = re.sub(r'(\$|\€|\£){1} ?\d+\.?\d* ?(m|bn|million|billion|trillion)?\b', ' [COVCURRENCY] ', text)
        text = re.sub(r'(\$|\€|\£){1} ?\d{1,3}((\,|\.)?\d{1,3}?)* ?(m|bn|million|billion|trillion)?\b', ' [COVCURRENCY] ', text)

        # - [COVDOSE] .. Numbers having mg, milligrams, g, grams
        # - [COVMEASURES] .. Numbers having cm, centimeters, m, meters, k, kilometers

        # Numbers having mg, milligrams, g, grams
        text = re.sub(r'\d{1,3}((\,|\.)?\d{1,3}?)* ?(mg|g|milligrams|grams)\b', ' [COVDOSE] ', text)

        # Numbers having cm, centimeters, m, meters, km, kilometers
        text = re.sub(r'\d{1,3}((\,|\.)?\d{1,3}?)* ?(mm|cm|m|km|millimeter(s)?|millimetre(s)?|centimeter(s)?|centimetre(s)?|meter(s)?|metre(s)?|kilometer(s)?|kilometre(s)?)\b', ' [COVMEASURES] ', text)
        text = re.sub(r'\d+(\,\d+)? ?feet', ' [COVMEASURES] ', text)
        text = re.sub(r'\d{1,2} ?inch(es)?', ' [COVMEASURES] ', text)

        # Time reference
        text = re.sub(r'(\d{1,2}\:\d{2}( ?(a\.?m\.?|p\.?m\.?))?( ?[a-zA-Z]{3,4}((\-|\+)\d{1,2})?)?)\b|(\d{1,2}( ?(a\.?m\.?|p\.?m\.?)){1}( ?[a-zA-Z]{3,4}((\-|\+)\d{1,2})?)?\b)', ' [COVTIME] ', text)
        text = re.sub(r'\d+(\,|\.)?(\d+)? ?(year(s)?|day(s)?|month(s)?|week(s)?)', ' [COVTIME] ', text)
        text = re.sub(r'\d+(\-| )?(hour(s)?|minute(s)?|second(s)?|millisecond(s)?)', ' [COVTIME] ', text)
        text = re.sub(r'\d+(\-| )?day', ' [COVTIME] ', text)

        # Date reference
        text = re.sub(r'((J|j)anuary|(F|f)ebruary|(M|m)arch|(A|a)pril|(M|m)ay|(J|j)une|(J|j)uly|(A|a)ugust|(S|s)eptember|(O|o)ctober|(N|n)ovember|(D|d)ecember|(J|j)an\.?|(F|f)eb\.?|(M|m)ar\.?|(A|a)pr\.?|(M|m)ay\.?|(J|j)un\.?|(J|j)ul\.?|(A|a)ug\.?|(S|s)ep\.?|(O|o)ct\.?|(N|n)ov\.?|(D|d)ec\.?) \d{1,2}(st|nd|rd|th)?\b(\,? (\d{4}))', ' [COVDATE] ', text)
        # text = re.sub(r'((J|j)anuary|(F|f)ebruary|(M|m)arch|(A|a)pril|(M|m)ay|(J|j)une|(J|j)uly|(A|a)ugust|(S|s)eptember|(O|o)ctober|(N|n)ovember|(D|d)ecember){1} \d{1,2}(st|nd|rd|th)?\b(\, (\d{4}))?', ' [COVDATE] ', text)
        text = re.sub(r'\d{1,2}(st|nd|rd|th)? ((J|j)anuary|(F|f)ebruary|(M|m)arch|(A|a)pril|(M|m)ay|(J|j)une|(J|j)uly|(A|a)ugust|(S|s)eptember|(O|o)ctober|(N|n)ovember|(D|d)ecember|(J|j)an\.?|(F|f)eb\.?|(M|m)ar\.?|(A|a)pr\.?|(M|m)ay\.?|(J|j)un\.?|(J|j)ul\.?|(A|a)ug\.?|(S|s)ep\.?|(O|o)ct\.?|(N|n)ov\.?|(D|d)ec\.?)( (\d{4}))', ' [COVDATE] ', text)
        text = re.sub(r'\d{1,2}(st|nd|rd|th)? ((J|j)anuary|(F|f)ebruary|(M|m)arch|(A|a)pril|(M|m)ay|(J|j)une|(J|j)uly|(A|a)ugust|(S|s)eptember|(O|o)ctober|(N|n)ovember|(D|d)ecember|(J|j)an\.?|(F|f)eb\.?|(M|m)ar\.?|(A|a)pr\.?|(M|m)ay\.?|(J|j)un\.?|(J|j)ul\.?|(A|a)ug\.?|(S|s)ep\.?|(O|o)ct\.?|(N|n)ov\.?|(D|d)ec\.?)\b', ' [COVDATE] ', text)
        text = re.sub(r'((J|j)anuary|(F|f)ebruary|(M|m)arch|(A|a)pril|(M|m)ay|(J|j)une|(J|j)uly|(A|a)ugust|(S|s)eptember|(O|o)ctober|(N|n)ovember|(D|d)ecember|(J|j)an\.?|(F|f)eb\.?|(M|m)ar\.?|(A|a)pr\.?|(M|m)ay\.?|(J|j)un\.?|(J|j)ul\.?|(A|a)ug\.?|(S|s)ep\.?|(O|o)ct\.?|(N|n)ov\.?|(D|d)ec\.?) \d{1,2}(st|nd|rd|th)?\b', ' [COVDATE] ', text)
        text = re.sub(r'((J|j)anuary|(F|f)ebruary|(M|m)arch|(A|a)pril|(M|m)ay|(J|j)une|(J|j)uly|(A|a)ugust|(S|s)eptember|(O|o)ctober|(N|n)ovember|(D|d)ecember|(J|j)an\.?|(F|f)eb\.?|(M|m)ar\.?|(A|a)pr\.?|(M|m)ay\.?|(J|j)un\.?|(J|j)ul\.?|(A|a)ug\.?|(S|s)ep\.?|(O|o)ct\.?|(N|n)ov\.?|(D|d)ec\.?)(\,? (\d{4}))\b', ' [COVDATE] ', text)
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
        text = re.sub(r'(\d{1,3}\-year\-old)|(\d{1,3} year(s)? old)', ' [COVAGE] ', text)
        text = re.sub(r'mid\-(\d{1}(0s){1})', ' [COVAGE] ', text)
        text = re.sub(r'\d{1}0{1}s{1}', ' [COVAGE] ', text)

        # Other semantics on covid numeric references
        text = re.sub(r'\d+((\,|\.)\d+)? ?(per( )?cent|\%)', ' [COVPERCENTAGE] ', text)

        # Coronavirus Deaths references
        text = re.sub(r'\d+((\,|\.)\d+)?( new)?( coronavirus)? death(s)?\b', ' [COVDEATHS] ', text)
        text = re.sub(r'\d+((\,|\.)\d+)* people( have)? died', ' [COVDEATHS] ', text)
        text = re.sub(r'\d+((\,|\.)\d+)* (fresh|more) death(s)?', ' [COVDEATHS] ', text)
        text = re.sub(r'death toll( of | at | )\d+((\,|\.)\d+)*', ' [COVDEATHS] ', text)
        text = re.sub(r'\d+((\,|\.)\d+)* people( have been)? killed', ' [COVDEATHS] ', text)
        text = re.sub(r'killing \d+((\,|\.)\d+)* people', ' [COVDEATHS] ', text)
        text = re.sub(r'killed( nearly)? \d+((\,|\.)\d+)* people', ' [COVDEATHS] ', text)

        # Coronavirus Tests references
        text = re.sub(r'\d+((\,|\.)\d+)?( new)?( coronavirus| covid)? test(s)?', ' [COVTESTS] ', text)
        text = re.sub(r'\d+((\,|\.)\d+)*( have| were)? tested', ' [COVTESTS] ', text)

        # Coronavirus Cases references
        text = re.sub(r'\d+((\,|\.)\d+)* ?(m|bn|million|billion|trillion)?( confirmed)?( coronavirus| covid| suspected| additional)? case(s)?', ' [COVCASES] ', text)
        text = re.sub(r'\d+((\,|\.)\d+)* ?(m|bn|million|billion|trillion)? new( confirmed)?( coronavirus| covid)?( confirmed)? case(s)?', ' [COVCASES] ', text)
        # TODO: Add more regex in COVCASES
        text = re.sub(r'\d+((\,|\.)\d+)* people( currently)? infected', ' [COVCASES] ', text)
        text = re.sub(r'infected( nearly)? \d+((\,|\.)\d+)*', ' [COVCASES] ', text)
        text = re.sub(r'\d+((\,|\.)\d+)*( new)? infections', ' [COVCASES] ', text)

        # Numbers references written in full-text
        text = re.sub(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirdy|forty|fifty|sixty|seventy|eighty|ninety|hundred(s)?|thousand(s)?|million(s)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\b', ' [COVNUMASWORD] ', text)

        # Telephone numbers
        # CY
        # Landlines
        text = re.sub(r'telephone(s)?(\:)?.{0,2}((\+|00)357 ?)?2\d ?\d{6}', ' [COVTELEPHONE] ', text)
        text = re.sub(r'(tel|call)(\:)?.{0,2}((\+|00)357 ?)?2\d ?\d{6}', ' [COVTELEPHONE] ', text)
        # Mobiles
        text = re.sub(r'(telephone|mobile)(s)?(\:)?.{0,2}((\+|00)357 ?)?9\d ?\d{6}', ' [COVTELEPHONE] ', text)
        text = re.sub(r'(tel|mob|call)(\:)?.{0,2}((\+|00)357 ?)?9\d ?\d{6}', ' [COVTELEPHONE] ', text)
        # IT
        # Landlines
        text = re.sub(r'telephone(s)?(\:)?.{0,2}((\+|00)39 ?)?0\d{1,2} ?3\d{3} ?\d{3} ?\d{4}', ' [COVTELEPHONE] ', text)
        text = re.sub(r'(tel|call)(\:)?.{0,2}((\+|00)39 ?)?0\d{1,2} ?3\d{3} ?\d{3} ?\d{4}', ' [COVTELEPHONE] ', text)
        # Mobiles
        text = re.sub(r'(telephone|mobile)(s)?(\:)?.{0,2}((\+|00)39 ?)?\d{2} ?\d{3} ?\d{4}', ' [COVTELEPHONE] ', text)
        text = re.sub(r'(tel|mob|call)(\:)?.{0,2}((\+|00)39 ?)?\d{2} ?\d{3} ?\d{4}', ' [COVTELEPHONE] ', text)
        # UK
        # Landlines/Mobiles
        text = re.sub(r'(telephone|mobile)(s)?(\:)?.{0,2}((\+|00)44 ?)?0?\d{4} ?\d{6}', ' [COVTELEPHONE] ', text)
        text = re.sub(r'(tel|mob|call)(\:)?.{0,2}((\+|00)44 ?)?0?\d{4} ?\d{6}', ' [COVTELEPHONE] ', text)
        # US
        # Landlines/Mobiles
        text = re.sub(r'(telephone|mobile)(s)?(\:)?.{0,2}((\+|00)1 ?)?\(?\d{3}\)? ?\d{3}( |-)?\d{4}', ' [COVTELEPHONE] ', text)
        text = re.sub(r'(tel|mob|call)(\:)?.{0,2}((\+|00)1 ?)?\(?\d{3}\)? ?\d{3}( |-)?\d{4}', ' [COVTELEPHONE] ', text)

        # Covid codename reference
        text = re.sub(r'covid-19', ' [COVCODENAME] ', text)
        text = re.sub(r'covid19', ' [COVCODENAME] ', text)
        text = re.sub(r'sars-cov-2', ' [COVCODENAME] ', text)
        text = re.sub(r'coronavirus', ' [COVCODENAME] ', text)

        # Number references in parentheses and brackets
        text = re.sub(r'\[\d+(\.{1,3}|\,)?\d*\]', ' [BRANUM] ', text)
        text = re.sub(r'\(\d+(\.{1,3}|\,)?\d*\)', ' [PARNUM] ', text)

        # Other general numeric references
        text = re.sub(r'\b\d{1,3}((\,|\.)?\d{1,3}?)*\b', ' [COVNUM] ', text)
        text = re.sub(r'\b\d+((\.|\,)\d+)*\b', ' [COVNUM] ', text)

        # Other number references fused in words
        text = re.sub(r'[a-zA-z]+\d+[a-zA-Z]*', '[WORDNUM]', text)
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

        # Replace 's occurrences with a single space
        text = re.sub(r'\'s', ' ', text)

        # Replace consequent punctuation with a single space
        text = re.sub(r'[`~!@#$%^&*()_=+\[\]{}\\\|;:\"\'<>.,/?]{2}', ' ', text)

        # Replace dots with a single space
        text = re.sub(r'\.', ' ', text)

        # Escape any other punctuation remains
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'u s', 'u_s', text)  # us acronym fix
    return text


# Apply a second round of cleaning
def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…—・]', '', text)
    text = re.sub('\n', ' ', text) #TODO: Check the applied fix from nothing to replacing with a space character [!]
    # Convert Dashes into Spaces
    text = re.sub('–', ' ', text)
    # Convert More than Two Spaces Sequences into one
    text = re.sub('(\s){2}(\s)*', ' ', text)
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
parser.add_argument("-optimize", default=False, help="Calculate perplexity and coherence scores to find the optimal "
                                                     "number of topics per given dataset",
                    dest='isOptEnabled', action='store_true')
parser.add_argument("-start", help="Starting number of topics to start the optimization process", type=int)
parser.add_argument("-step", help="A step value for increasing the number of topics during the optimization process",
                    type=int)
parser.add_argument("-limit", help="Limit number of topics to stop the optimization process",
                    type=int)
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

print('■■■ Lowercase transformation..', end='')
round_lw = lambda x: clean_lowercase(x)

data_clean = pd.DataFrame(data_df.Content.apply(round_lw))
print('Done')

# Todo: Latest change: test and remove >>>
# print('■ Removing English Stop-Words..', end='')
# cv = CountVectorizer(stop_words='english')
# data_cv = cv.fit_transform(docs)
# data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names(), index=headlines)
# print('Done')
# <<<
# print(data_dtm)

# STEP 2: Cleaning Step
print('■ Cleaning data contents..')

# -->>> Additional cleaning steps for numeric references and code snippets cleaning

print('■■■ Marking code snippets..', end='')
round_code = lambda x: clean_code(x)

data_clean = pd.DataFrame(data_clean.Content.apply(round_code))
print('Done')

print('■■■ Cleaning pre-defined exclusions..', end='')
round_exclusions = lambda x: clean_exclusions(x)

data_clean = pd.DataFrame(data_clean.Content.apply(round_exclusions))
print('Done')

print('■■■ Marking numeric references..', end='')
round_nums = lambda x: clean_numbers(x)

data_clean = pd.DataFrame(data_clean.Content.swifter.apply(round_nums))
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

print('■■■ Grouping pre-defined tokens..', end='')
round_grouping = lambda x: clean_word_grouping(x)

data_clean = pd.DataFrame(data_clean.Content.apply(round_grouping))
print('Done')

print('■ Removing English Stop-Words..', end='')
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(docs)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names(), index=headlines)
print('Done')

# If more than half of the media have it as a top word, exclude it from the list (optional step for later)
add_stop_words = ['said', 'åêåêåê']

# Add new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# STEP 3: Generate Document-Term Matrix
# Create a document-term matrix using CountVectorizer, and exclude common English stop words
print('■ Generating Document-Term Matrix using sklearn..', end='')
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.Content)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names(), index=headlines)
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

def run_lda(corpus, id2word, topics_num, passes_num, alpha, enable_messages=True):
    # Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),
    # we need to specify two other parameters as well - the number of topics and the number of passes
    #    Note for the LDA parameters: Use the following if you want to run LDA with fixed seed(s) or compute other stuff:
    #    random_state=100, update_every=1, chunksize=100, per_word_topics=True
    if enable_messages:
        print('■ Latent Dirichlet Allocation (LDA) analysis is loading. Please wait..', end='')
    lda_model = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=topics_num, passes=passes_num, alpha=alpha)

    return lda_model


def compute_percoh_values(corpus, dictionary, passes, alpha, limit, start=2, step=1):
    """
    Compute perplexity and u_mass coherence for various number of topics

    Parameters:
    ----------
    corpus : Gensim corpus
    dictionary : Gensim dictionary
    passes : Number of passes for the LDA execution
    alpha : Alpha hyper-parameter value for the LDA execution
    limit : Max num of topics
    start : Starting value of topics to compute for the optimization process
    step : Step value to increase the number of topics during the optimization process

    Returns:
    -------
    model_list : List of LDA topic models
    perplexity_values : Perplexity values corresponding to the LDA model with respective number of topics
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    perplexity_values = []
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print('■■ Running LDA for ' + str(num_topics) + ' topics in ' + str(passes_num) + ' passes..', end='')
        lda_model = run_lda(corpus, dictionary.id2token, num_topics, passes, alpha, enable_messages=False)
        # model_list.append(lda_model)

        # Compute Perplexity
        perplexity_score = lda_model.log_perplexity(corpus)  # a measure of how good the model is. lower the better.
        perplexity_values.append(perplexity_score)

        # Compute Coherence Score

        # coherence_model_lda = CoherenceModel(model=lda, texts=corpus, dictionary=d, coherence='c_v')
        coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
        coherence_score = coherence_model_lda.get_coherence()

        coherence_values.append(coherence_score)

        print('Done')

    return model_list, perplexity_values, coherence_values


def save_score(values, save_as):
    with open('output/' + save_as + '.txt', 'w') as f:
        for item in values:
            f.write("%s\n" % item)
        f.close()
    return True

if args.isOptEnabled:
    print('■ LDA Optimization Process is starting..')

    start_from = 2
    step = 1
    limit = 20

    if args.start:
        start_from = args.start
    if args.step:
        step = args.step
    if args.limit:
        limit = args.limit

    # Loop for increasing the number of topics and calculating the scores
    # cur_num_of_topics = start_from
    # while cur_num_of_topics <= limit:
    #     lda = run_lda(corpus, id2word, cur_num_of_topics, passes_num, 'auto')
    #

    # Construct Dictionary
    word2id = dict((k, v) for k, v in cv.vocabulary_.items())
    d = corpora.Dictionary()
    d.id2token = id2word
    d.token2id = word2id
    #
    #     cur_num_of_topics += step
    model_list, perplexity_values, coherence_values = compute_percoh_values(corpus=corpus, dictionary=d,
                                                                            passes=passes_num, alpha='auto',
                                                                            limit=limit, start=start_from, step=step)

    print('■■ Generating plots for perplexity and coherence scores..', end='')

    # Show graph for Coherence Score
    x = range(start_from, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    # plt.show()
    plt.savefig('output/coh_' + os.path.basename(pkl_file).replace('.pkl', '') + '.png')
    plt.close()

    # Show graph for Perplexity Score
    x = range(start_from, limit, step)
    plt.plot(x, perplexity_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Perplexity score")
    plt.legend(("perplexity_values"), loc='best')
    # plt.show()
    plt.savefig('output/per_' + os.path.basename(pkl_file).replace('.pkl', '') + '.png')
    plt.close()

    print('Done')

    print('■■ Saving list of number of topics values to disk..', end='')
    num_of_topics_values = []
    for num in range(start_from, limit, step):
        num_of_topics_values.append(num)

    save_score(values=num_of_topics_values, save_as='num_of_topics_values')
    print('Done')

    print('■■ Saving list of perplexity values to disk..', end='')
    save_score(values=perplexity_values, save_as='perplexity_values')
    print('Done')

    print('■■ Saving list of coherence values to disk..', end='')
    save_score(values=coherence_values, save_as='coherence_values')
    print('Done')

    print('Optimization process finished. Exiting..')

    exit()
else:
    lda = run_lda(corpus, id2word, topics_num, passes_num, 'auto')

    print('Done')
    print('\n** ' + str(topics_num) + ' topics found in ' + str(passes_num) + ' passes:\n')
    print(lda.print_topics())

    # Compute Perplexity
    print('\nPerplexity: ', lda.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    word2id = dict((k, v) for k, v in cv.vocabulary_.items())
    # Debugging:
    # print(word2id)
    d = corpora.Dictionary()
    d.id2token = id2word
    d.token2id = word2id

    # coherence_model_lda = CoherenceModel(model=lda, texts=corpus, dictionary=d, coherence='c_v')
    coherence_model_lda = CoherenceModel(model=lda, corpus=corpus, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


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
