import pandas as pd
import numpy as np
import time
import os
import sys
sys.path.append('secrets_local/')
import secrets_local
import xml.etree.ElementTree as et
from lxml import etree
import requests
import re
import dateutil.parser
from pymarc import MARCReader
import csv
import io
from bs4 import BeautifulSoup
import glob
sys.path.append('scripts/')
from functions import *

oDir = "./Output"
if not os.path.isdir(oDir) or not os.path.exists(oDir):
    os.makedirs(oDir)

pDir = "./Processing"
if not os.path.isdir(pDir) or not os.path.exists(pDir):
    os.makedirs(pDir)

aDir = "./Analysis"
if not os.path.isdir(aDir) or not os.path.exists(aDir):
    os.makedirs(aDir)

####NLM import ln#22-168


## 25-287 ingest Docline data from MARC binary data,
## either from a downloaded file or from the latest
## serials MARC data from the NLM Locator catalog
##
## and also ingest Alma Analytics data, either from downloaded
## file or via API at target set in configs

nlm_choice = input("Do you want to have the program ingest the latest NLM from the web, or do you have a local copy in the 'NLM' folder you would like it to use?  Type the number next to your choice.\n\t1-Retrive for me\n\t2-I have a local copy\n\t3-I've processed it already and it's in Processing/journal_data.csv\nChoice:")

input("Download your local version of the Analytics all journals in the format at \"/shared/Community/Reports/Institutions/Tufts University\".  Put this in the \"Analytics/\" folder in this directory.\nPress any key to continue")
#analytics_choice = input("Do you want to have the program ingest the your journal data from Analytics, or do you have a local copy in the 'Analytics' folder you would like it to use?  Note that you'll have to put the path in your secrets_local file if 1.  \nType the number next to your choice.\n\t1-Retrive for me\n\t2-I have a local copy\nChoice:")
print_or_electronic_choice = input("Do you want to parse/compare print, electronic, or both?\n\t1-Print\n\t2-Electronic\n\t3-Both\nChoice:")
print_or_electronic_choice = str(print_or_electronic_choice)
input("The Docline file will be retrieved from the \"Docline\" folder in this directory.   Retrieve this from your Docline institutional account\nPress any key to continue")
#
#

# get MARC data

getNLMData(nlm_choice)
# get Analytics data
analytics_filename = getAnalyticsData()

#group and merge
merged_df = groupAndMergeAlmaAndDocline(analytics_filename)



# convert Alma medical journals data into Docline format
files_docline = glob.glob('Docline/*', recursive = True)
docline_filename = files_docline[0]
docline_df = pd.read_csv(docline_filename, engine='python')
alma_nlm_merge_df = convert(merged_df, docline_df)


# prepare Docline and Alma Medical journals CSV (Docline-formatted)
# for merging

return_list = prepare(alma_nlm_merge_df, docline_df, print_or_electronic_choice)

alma_nlm_merge_df = return_list[0]
existing_docline_df = return_list[1]

merge(alma_nlm_merge_df, existing_docline_df)


sys.exit()
#full_match_output_df[full_match_output_df[['record_type'] == 'RANGE']] =
# old_ranges_df = old_ranges_df[old_ranges_df.nlm_unique_id.isin(different_docline_range_compare_df.nlm_unique_id)]
#
# full_matched_output_df = matched_nlm_df.copy()
#current_alma_df[~current_alma_df.index.isin(deleted_alma_df.index)]
# full_matched_output_df = full_matched_output_df[full_matched_output_df.nlm_unique_id.isin(full_matched_df.nlm_unique_id)]
#
#
#
# changed_matched_df = changed_matched_output_df[~changed_matched_output_df.nlm_unique_id.isin(full_matched_df.nlm_unique_id)]

# reformate valid dataframe content into Docline format, where holding data isnt repeated


old_ranges_df['action'] = 'DELETE'

changed_matched_df['action'] = 'ADD'

pd.set_option('display.max_columns', None)
print("old ranges")
print(old_ranges_df)

old_ranges_df.to_csv('Output/Output from Docline for Delete.csv', index=False)


print("updated_ranges")
print(changed_matched_df)
changed_matched_df.to_csv('Output/New Ranges from Alma.csv', index=False)

print("Alma adds")
print(add_df)
add_df.to_csv('Output/Total New Records from Alma.csv', index=False)

print("Deletes to take out of Docline")
print(deleted_output_df)
deleted_output_df.to_csv('Output/Deleted From Alma to Take Out of Docline.csv', index=False)






alma_nlm_csv = open("Alma Docline output.csv", encoding='utf-8')
docline_csv = open('Existing Docline Holdings.csv', encoding='utf-8')

fifth_time = time.time()
log_file.write("--- %s execution time of parsing output ---\n" % ((fifth_time - fourth_time)/60))
log_file.write("--- %s execution time of whole program ---\n" % ((fifth_time - start_time)/60))
print("--- %s execution time of parsing output ---\n" % ((fifth_time - fourth_time)/60))
print("--- %s execution time of whole program ---\n" % ((fifth_time - start_time)/60))

log_file.close()
sys.exit()
with open(alma_nlm_csv, 'r') as file1:
    with open(docline_csv, 'r') as file2:
        with open('Tufts Docline Upload.csv', 'w') as outfile:
            writer = csv.writer(outfile)
            reader1 = csv.reader(file1)
            reader2 = csv.reader(file2)

            for row in reader1:
                if not row:
                    continue

                for other_row in reader2:
                    if not other_row:
                        continue

                    # if we found a match, let's write it to the csv file with the id appended
                    if row[1].lower() == re.sub(r'NLM_(\d+)', r'\1', other_row[4].lower()):
                        new_row = other_row
                        new_row.append(row[0])
                        writer.writerow(new_row)
                        continue

                # reset file pointer to beginning of file
                file2.seek(0)
# print(type(alma_nlm_merge_df.loc[2, 'nlm_unique_id']))
# print(type(alma_nlm_merge_df.loc[2, 'begin_year']))
# print(type(alma_nlm_merge_df.loc[2, 'begin_volume']))
# # Convert final rows

#print("--- %s execution time of docline file write ---\n" % ((fourth_time - third_time)/60))


#log_file = open("Execution time of docline script.txt", "w+")
