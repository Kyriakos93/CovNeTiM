""" Duplicates Remover

Removes the duplicated (identical) or similar articles that refer to an identical or similar news headline and
generates a new CSV file free from duplications.

Usage: python3 duplicates_remover.py -file [input_filepath] -export [output_filepath] -threshold 0.80
Note: It's recommended to use the similarity of 0.80 seems to refer to the same article and better perform

"""
import argparse

import pandas as pd
import textdistance

filepath = None
similarity_threshold = 0.80 # default
export_path = None

print(
'________               .__  .__               __                  __________                                         \n'+
'\______ \  __ ________ |  | |__| ____ _____ _/  |_  ____   ______ \______   \ ____   _____   _______  __ ___________ \n'+
' |    |  \|  |  \____ \|  | |  |/ ___\\\\__  \\\\   __\/ __ \ /  ___/  |       _// __ \ /     \ /  _ \  \/ // __ \_  __ \\\n'+
' |    `   \  |  /  |_> >  |_|  \  \___ / __ \|  | \  ___/ \___ \   |    |   \  ___/|  Y Y  (  <_> )   /\  ___/|  | \/\n'+
'/_______  /____/|   __/|____/__|\___  >____  /__|  \___  >____  >  |____|_  /\___  >__|_|  /\____/ \_/  \___  >__|   \n'+
'        \/      |__|                \/     \/          \/     \/          \/     \/      \/                 \/       \n')

print("Welcome to RISE TAG Duplicates Remover v1.0\n")

parser = argparse.ArgumentParser()
parser.add_argument("-file", help="Input CSV file to clean it from duplicated or similar records", type=str)
parser.add_argument("-export", help="Output CSV file name of the new dataset", type=str)
parser.add_argument("-threshold", help="Decimal similarity threshold (e.g. 0.80)", type=float)
args = parser.parse_args()

if args.file:
    filepath = args.file
else:
    print('[!] Input file path not provided. Please provide the valid options to run the script.'
          '\nDuplicate Remover is terminated.')
    exit()

if args.export:
    export_path = args.export
else:
    print('[!] Output file path not provided. Please provide the valid options to run the script.'
          '\nDuplicate Remover is terminated.')
    exit()

if args.threshold:
    similarity_threshold = args.threshold

# Read CSV File
print('■ Reading CSV File (' + filepath + ') into DataFrame..', end='')
data_df = pd.read_csv(filepath, low_memory=False)
print('Done')

# Loaded records
index = data_df.index
number_of_rows = len(index)

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

print('╔═══════════════════════════════════════════════════╗')
print('║ Threshold Applied: ' + str(similarity_threshold) + ' \t\t\t\t\t\t\t║')
print('║ Similar Records Dropped: ' + str(droped_records) + ' out of ' + str(number_of_rows) + ' loaded \t║')
print('║ Remaining Records in Dataset: ' + str(len(data_df.index)) + ' \t\t\t\t║')
print('╚═══════════════════════════════════════════════════╝')

if '.csv' not in export_path:
    export_path += '.csv'
print('■ Exporting processed CSV file: ' + export_path + '.. ', end='')
data_df.to_csv(export_path)
print('Done')

print('\nDuplicates Remover finished processing. Exiting..')
exit()
