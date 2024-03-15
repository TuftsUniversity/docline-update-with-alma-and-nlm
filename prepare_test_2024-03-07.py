import glob
from bs4 import BeautifulSoup
import io
import csv
from pymarc import MARCReader
import dateutil.parser
import re
import requests
from lxml import etree
import xml.etree.ElementTree as et
import sys
sys.path.append('secrets_local/')
import secrets_local
import pandas as pd
import numpy as np
import time
import os
import re

def convert(merged_df, docline_df, choice):


    # print(merged_df)
    # sys.exit()
    alma_nlm_merge_df = pd.DataFrame(columns=docline_df.columns)

    alma_nlm_merge_df['Lifecycle'] = ""

    #alma_nlm_merge_df['embargo_period'] = alma_nlm_merge_df['embargo_period'].fillna(0)

    merged_df['ISSN_x'] = merged_df['ISSN_x'].astype(str)



    #start_time = time.time()
    #second_time = time.time()
    #third_time = time.time()
    #print("--- %s minutes to complete merging ---\n" % ((third_time - second_time)/60))

    #log_file.write("--- %s minutes to complete merging ---\n" % (third_time - second_time))


    # Iterate through each row in the merged DataFrame to create the necessary records
    # In this creation of your institution's medical journal data in Docline format,
    # I'm adding the additional column "Bibliographic Lifecycle", which I will later
    # take out, because this will help determine what needs to be deleted

    # create docline-formatted output from Alma data

    # this section was made with the help of ChatGPT 4,
    # just for reference
    x = 0
    for idx, row in merged_df.iterrows():
        if x % 100 == 0:
            print(str(row['Title_x']) + "--" + str(row['ISSN_x']))
        x += 1
        # Create the main HOLDING row with metadata
        # embargo_period = 0
        #
        # if choice == "1":
        #     embargo_period = 0
        # elif choice == "2":
        #     if row['Embargo Months'] is not None:
        #         embargo_period = row['Embargo Months']
        #         # print("Embargo Months")
        #         # print(embargo_period)
        #
        #     elif row['Embargo Years'] is not None:
        #         embargo_period = row['Embargo Months'] * 12
        #         # print("Embargo Years")
        #         # print(embargo_period)
        #
        #
        #     else:
        #         embargo_period = 0

        main_row = {
            'Bibliographic Lifecycle': row['Lifecycle'],
            'action': '',
            'record_type': 'HOLDING',
            'libid': secrets_local.libid,
            'serial_title': row['Title_x'],
            'nlm_unique_id': row['NLM_Unique_ID'],
            'holdings_format': row['Electronic or Physical'],
            'begin_volume': None,
            'end_volume': None,
            'begin_year': '',
            'end_year': '',
            'issns': row['ISSN_x'].replace(';', ','),
            'currently_received': 'No' if 'until' in str(row['Coverage Information Combined']) else 'Yes',
            'retention_policy': secrets_local.retention_policy,
            'limited_retention_period': secrets_local.limited_retention_period,
            'embargo_period': 0,#embargo_period, #0 if choice == 1 elif row['Embargo Months']#secrets_local.embargo_period,
            'limited_retention_type': secrets_local.limited_retention_type,
            'has_epub_ahead_of_print': secrets_local.has_epub_ahead_of_print,
            'has_supplements': secrets_local.has_supplements,
            'ignore_warnings': secrets_local.ignore_warnings,
            'last_modified': ''#pd.to_datetime('today').strftime('%Y-%b-%d')
        }
        alma_nlm_merge_df = pd.concat([alma_nlm_merge_df, pd.DataFrame(main_row, index=[0])])

        # Create additional HOLDING rows for coverage data
        coverage_data = re.sub(r';{2,}', r';', row['Coverage Information Combined'])
        coverage_data = str(coverage_data).split(';')

        embargo_months = str(row['Embargo Months']).split(',')

        # embargo_months = embargo_months.replace('[', '')
        # embargo_months = embargo_months.replace(']', '')
        # embargo_years = embargo_years.replace('[', '')
        # embargo_years = embargo_years.replace(']', '')
        embargo_years = str(row['Embargo Years']).split(',')
        if x % 100 == 0:
            print(str(row['Title_x']) + "--" + str(row['ISSN_x']))

        x += 1

        coverage_data_output = []
        for coverage, month, year in zip(coverage_data, embargo_months, embargo_years):
            embargo_period = 0
            begin_year = None
            end_year = None
            begin_volume = None
            end_volume = None
            month = str(month).replace('[', '')
            month = str(month).replace(']', '')
            year = str(year).replace('[', '')
            year = str(year).replace(']', '')
            month = int(float(month))
            year = int(float(year))
            if choice == "1":
                embargo_period = 0
            elif choice == "2":
                if month is not None:

                    embargo_period = month
                    # print("Embargo Months")
                    # print(embargo_period)

                elif year is not None:
                    embargo_period = year * 12
                    # print("Embargo Years")
                    # print(embargo_period)


            else:
                embargo_period = 0

            if 'from' in coverage and 'until' in coverage:
                beginning = coverage.split('from')[-1].split('until')[0].strip()#.split(r'[-\s \\]')[-1].strip()

                begin_year = re.sub(r'.*?(\d{4}).*$', r'\1', beginning)

                if 'volume' in beginning:
                    begin_volume = re.sub(r'    .*?volume\:\s+(\d+).*$', r'\1', beginning)


                end = coverage.split('until')[-1]

                end_year = re.sub(r'.*?(\d{4}).*$', r'\1', end)

                # if 'volume' in end:
                #     end_volume = re.sub(r'    .*?volume\:\s+(\d+).*$', r'\1', end)

            elif 'from' in coverage:

                beginning = coverage.split('from')[-1].strip()
                begin_year = re.sub(r'.*?(\d{4}).*$', r'\1', beginning)
                # if 'volume' in beginning:
                #     begin_volume = re.sub(r'    .*?volume\:\s+(\d+).*$', r'\1', beginning)
                # else:
                #     begin_volume = None



            coverage_row = {
                'Bibliographic Lifecycle': row['Lifecycle'],
                'action': '',
                'record_type': 'RANGE',
                'libid': secrets_local.libid,
                'serial_title': row['Title_x'],
                'nlm_unique_id': row['NLM_Unique_ID'],
                'holdings_format': row['Electronic or Physical'],
                'begin_volume': None,
                'end_volume': None,
                'begin_year': begin_year,
                'end_year': end_year,
                'issns': row['ISSN_x'].replace(';', ','),
                'currently_received': 'No' if 'until' in str(row['Coverage Information Combined']) else 'Yes',
                'retention_policy': secrets_local.retention_policy,
                'limited_retention_period': secrets_local.limited_retention_period,
                'embargo_period': embargo_period,#secrets_local.embargo_period,
                'limited_retention_type': secrets_local.limited_retention_type,
                'has_epub_ahead_of_print': secrets_local.has_epub_ahead_of_print,
                'has_supplements': secrets_local.has_supplements,
                'ignore_warnings': secrets_local.ignore_warnings,
                'last_modified': ''
            }
            coverage_data_output.append(coverage_row)

        updated_set = set()
        new_coverage_data_output = []
        for d in coverage_data_output:

            t = tuple(d.items())
            if t not in updated_set:
                updated_set.add(t)
                new_coverage_data_output.append(d)

        coverage_data_output = [dict(t) for t in {tuple(d.items()) for d in coverage_data_output}]

        for output_row in new_coverage_data_output:
            alma_nlm_merge_df = pd.concat([alma_nlm_merge_df, pd.DataFrame(output_row, index=[0])])

    return alma_nlm_merge_df

merged_deduped_all_df = pd.read_excel('Processing/Book2.xlsx', engine="openpyxl")
docline_df = pd.read_csv('Docline/library_holdings_MAUTUF_2024-02-22.csv', engine='python')

output_df = convert(merged_deduped_all_df, docline_df, "2")

output_df.to_csv('Processing/test.csv', index=False)
