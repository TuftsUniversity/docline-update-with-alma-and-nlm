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
import re


# def parse_xml_chunked(file_path, tag):
#     for event, elem in ET.iterparse(file_path, events=("start", "end")):
#         if event == "end" and elem.tag == tag:
#             yield elem
#             elem.clear()
def getNLMData(nlm_choice):

    #43-366
    start_time = time.time()
    filename = ""
    if str(nlm_choice) == "1":
        webpage = requests.get("https://ftp.nlm.nih.gov/projects/serfilelease")

        soup = BeautifulSoup(webpage.content, "html.parser")
        dom = etree.HTML(str(soup))
        # this loops through the links in this directory
        # the last link will be the one that is assigned
        # to the variable a in the last iteration of the loop
        for a in dom.xpath('//ul/li/a/@href'):

            if re.match(r'.xml$', a):
                continue
            else:
                a = a

        latest_file = a


        full_nlm_serials_data_mrc = requests.get("https://ftp.nlm.nih.gov/projects/serfilelease/" + str(latest_file))
        filename = "marc_output_via_bs4_and_requests.mrc"
        full_filename = "NLM/" + filename
        mrc_output = open("NLM/" + filename + ".mrc", "wb")
        mrc_output.write(full_nlm_serials_data_mrc.content)
    elif nlm_choice == "2":

        files = glob.glob('NLM/*.mrc', recursive = True)

        full_filename = files[0]







    if nlm_choice == "1" or nlm_choice == "2":

        #open text file in read mode

        csv_data = []
        csv_headers = ["Title", "NLM_Unique_ID", "OCLC_Number", "Print_ISSN", "Electronic_ISSN", "ISSN"]

        nlm_df = pd.DataFrame(columns=csv_headers)
        with open(full_filename, 'rb') as fh:
            reader = MARCReader(fh, to_unicode=True, force_utf8=True)
            x = 0

            for record in reader:

                #print(record['245']['a'])

                title = record['245']['a']
                title = title.encode('utf-8').decode()
                nlm_unique_id = record['001'].value()
                oclc_number = "None"
                print_issn = "None"
                electronic_issn = "None"
                issn = "None"


                # print("NLM Unique ID" + nlm_unique_id)
                for oclc_035_tag in record.get_fields('035'):

                    dict = oclc_035_tag.subfields_as_dict()
                    oclc_list = []
                    for key, value in dict.items():

                        for each_value in value:

                            if re.search("\(OCoLC\)", each_value):
                                oclc_list.append(re.sub(r'\(OCoLC\)\s*(\d+)', r'\1', each_value))

                    oclc_list = list(dict.fromkeys(oclc_list))
                    oclc_number = ";".join(oclc_list)


                try:
                    for record_022 in record.get_fields('022'):
                        issn = record_022['a']

                        print_or_electronic = "None"
                        try:
                            print_or_electronic = record_022['7']
                        except:
                            print_or_electronic = "None"
                        if 'Print' in print_or_electronic:
                            print_issn = record_022['a']
                        elif 'Electronic' in print_or_electronic:
                            electronic_issn = record_022['a']
                        else:
                            print_or_electronic = "None"
                            issn = record_022['a']
                except:
                    issn = None


                csv_data.append([title, nlm_unique_id, oclc_number, print_issn, electronic_issn, issn])
                nlm_df = pd.concat([nlm_df, pd.DataFrame({"Title": title, "NLM_Unique_ID": nlm_unique_id, "OCLC_Number": oclc_number, "Print_ISSN": print_issn, "Electronic_ISSN": electronic_issn, "ISSN": issn}, index=[0])])
                if x % 1000 == 0:
                    print(x)

                    # print(csv_data[x])
                x += 1



        csv_filename = "Processing/journal_data.csv"
        with open(csv_filename, encoding='utf-8', mode="w", newline="\n") as csv_file:
            writer = csv.writer(csv_file, delimiter='\t')
            writer.writerow(csv_headers)
            writer.writerows(csv_data)

        #

def getAnalyticsData():
    print("get analytics data")
    files = glob.glob('Analytics/*.csv', recursive = True)
    #
    analytics_filename = files[0]


    return analytics_filename

def groupAndMergeAlmaAndDocline(analytics_filename, choice):
    df = pd.read_csv(analytics_filename, dtype={"MMS Id": "str", 'Network Number (OCoLC)': 'str'}, encoding='utf-8')


    df['Title'] = df['Title'].apply(lambda x: re.sub(r'\.$', r'', x))
    df['Title'] = df['Title'].apply(lambda x: re.sub(r'^(.+?)(\s\:|$)', r'\1', x))

    df['Embargo Months'] = df['Embargo Months'].fillna(0)
    df['Embargo Years'] = df['Embargo Years'].fillna(0)

    print("read")

    # 1. group by
    # print(df)
    if choice == "1":
        grouped_df = df.groupby(["Title", "MMS Id", "Network Number (OCoLC)", "ISSN", "Lifecycle", "Electronic or Physical"])["Coverage Information Combined"].apply(lambda x: ';'.join(x)).reset_index()
    elif choice == "2":

        grouped_df = df.groupby(["Title", "MMS Id", "Network Number (OCoLC)", "ISSN", "Lifecycle", "Electronic or Physical"], dropna=False).agg({"Coverage Information Combined": lambda x: ';'.join(x), 'Embargo Months': lambda x: str(x.tolist()), 'Embargo Years': lambda x: str(x.tolist())}).reset_index()
    else:
        grouped_df = df.groupby(["Title", "MMS Id", "Network Number (OCoLC)", "ISSN", "Lifecycle", "Electronic or Physical"])["Coverage Information Combined"].apply(lambda x: ';'.join(x)).reset_index()
    print("complete groupby")
    # print(grouped_df)



    # 1. Explode on OCLC

    exploded_oclc_df = grouped_df.copy()
    exploded_oclc_df['OCLC'] = exploded_oclc_df['Network Number (OCoLC)'].str.split(';')
    #                                            Network Number (OCoLC)
    exploded_oclc_df = exploded_oclc_df.explode('OCLC').reset_index(drop=True)
    # print(exploded_oclc_df)


    exploded_oclc_df['OCLC'] = exploded_oclc_df['OCLC'].str.strip()
    pd.set_option('display.max_columns', None)
    # print(exploded_oclc_df.head(100))

    exploded_oclc_df['OCLC'] = exploded_oclc_df['OCLC'].astype('string')


    analytics_ocolc_df = exploded_oclc_df[exploded_oclc_df['OCLC'].str.contains("OCoLC")]


    analytics_no_ocolc_df = exploded_oclc_df[~exploded_oclc_df['OCLC'].notna()]







    # 2. explode Analytics on ISSN
    # this is a separate set than the OCLC set above, so the
    # merge will be done separately and then concatenated

    exploded_issn_df = grouped_df.copy()
    exploded_issn_df['ISSN'] = exploded_issn_df['ISSN'].str.split('; ')
    exploded_issn_df = exploded_issn_df.explode('ISSN').reset_index(drop=True)


    exploded_issn_df['ISSN'] = exploded_issn_df['ISSN'].str.strip()


    exploded_issn_df = exploded_issn_df[exploded_issn_df['ISSN'].notna() & exploded_issn_df['ISSN'] != 'None']



    # 3. Merge with NLM data, mostly to get NLM Unique ID for the Alma data
    journal_data_df = pd.read_csv("Processing/journal_data.csv", delimiter="\t")



    print("complete read csv")

    exploded_oclc_journal_data_df = journal_data_df.copy()
    exploded_oclc_journal_data_df = exploded_oclc_journal_data_df.copy()
    exploded_oclc_journal_data_df['OCLC_Number'] = exploded_oclc_journal_data_df['OCLC_Number'].str.split(';')
    exploded_oclc_journal_data_df = exploded_oclc_journal_data_df.explode('OCLC_Number')
    exploded_oclc_journal_data_df['OCLC_Number'] = exploded_oclc_journal_data_df['OCLC_Number'].str.strip()

    pd.set_option('display.max_columns', None)

    exploded_oclc_journal_data_df = exploded_oclc_journal_data_df[exploded_oclc_journal_data_df['OCLC_Number'].notna() & exploded_oclc_journal_data_df['OCLC_Number'] != 'None']


    exploded_oclc_journal_data_df['OCLC_Number'] = exploded_oclc_journal_data_df['OCLC_Number'].astype("string")
    exploded_oclc_journal_data_df['OCLC_Number'] = exploded_oclc_journal_data_df['OCLC_Number'].apply(lambda x: re.sub(r'\(OCoLC\)[A-Za-z\s]*(\d+)', r'\1', str(x)))




    log_file = open("Output/Execution time of docline script.txt", "w+")

    # 4. restrict NLM subset to those that have ISSN
    # as noted above for the Alma data, this is a separate set
    # that will be matched separately and then
    # concatenated together with the OCLC matches

    issn_journal_data_df = journal_data_df.copy()





    issn_journal_data_df = issn_journal_data_df.loc[((issn_journal_data_df['ISSN'] != "None") | (issn_journal_data_df['Electronic_ISSN'] != "None") | (issn_journal_data_df['Print_ISSN'] != "None"))]





    # 5. merge
    # this merges on matches, and then concatenates all the merges
    # together into a single dataframe that can go
    # into the following Docline parsing and
    # comparison processes


    merged_oclc_df = analytics_ocolc_df.merge(exploded_oclc_journal_data_df, left_on='OCLC', right_on="OCLC_Number", how='inner')

    merged_oclc_df = merged_oclc_df.rename(columns={"ISSN_y": 'ISSN'})


    exploded_issn_df.to_csv("Processing/Exploded Analytics ISSN.csv", index=False)
    issn_journal_data_df.to_csv("Processing/Exploded NLM ISSN.csv", index=False)
    issn_journal_data_df = issn_journal_data_df.dropna(subset=['ISSN'])





    merged_issn_df = exploded_issn_df[pd.notnull(exploded_issn_df['ISSN'])].merge(issn_journal_data_df[pd.notnull(issn_journal_data_df.ISSN)], on='ISSN', how='inner')

    merged_electronic_issn_df = exploded_issn_df[pd.notnull(exploded_issn_df['ISSN'])].merge(issn_journal_data_df[pd.notnull(issn_journal_data_df.Electronic_ISSN)], left_on='ISSN', right_on="Electronic_ISSN", how='inner')
    merged_print_issn_df = exploded_issn_df[pd.notnull(exploded_issn_df['ISSN'])].merge(issn_journal_data_df[pd.notnull(issn_journal_data_df.Print_ISSN)], left_on='ISSN', right_on="Print_ISSN", how='inner')


    merged_df = merged_oclc_df.copy()

    merged_issn_df = pd.concat([merged_issn_df, merged_electronic_issn_df, merged_print_issn_df])
    merged_df = pd.concat([merged_oclc_df, merged_issn_df, merged_electronic_issn_df]).reset_index()

    merged_df = merged_df.drop('ISSN_y', axis=1)


    merged_df = merged_df.drop_duplicates(subset=['NLM_Unique_ID', 'Electronic or Physical'])


    # 6. Return the merged data


    return merged_df

def convert(merged_df, docline_df, choice):



    alma_nlm_merge_df = pd.DataFrame(columns=docline_df.columns)

    alma_nlm_merge_df['Lifecycle'] = ""




    merged_df['ISSN_x'] = merged_df['ISSN_x'].fillna("").astype(str)


    x = 0
    for idx, row in merged_df.iterrows():
        if x % 500 == 0:
            print(str(row['Title_x']) + "--" + str(row['ISSN_x']))
        x += 1

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


        embargo_years = str(row['Embargo Years']).split(',')
        # if x % 100 == 0:
        #     print(str(row['Title_x']) + "--" + str(row['ISSN_x']))

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


                elif year is not None:
                    embargo_period = year * 12



            else:
                embargo_period = 0

            if 'from' in coverage and 'until' in coverage:
                beginning = coverage.split('from')[-1].split('until')[0].strip()#.split(r'[-\s \\]')[-1].strip()

                begin_year = re.sub(r'.*?(\d{4}).*$', r'\1', beginning)

                if 'volume' in beginning:
                    begin_volume = re.sub(r'    .*?volume\:\s+(\d+).*$', r'\1', beginning)


                end = coverage.split('until')[-1]

                end_year = re.sub(r'.*?(\d{4}).*$', r'\1', end)

            elif 'from' in coverage:

                beginning = coverage.split('from')[-1].strip()
                begin_year = re.sub(r'.*?(\d{4}).*$', r'\1', beginning)



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

def propagate_nlm_unique_id_and_libid_values(df_d):
    columns_to_update = ['nlm_unique_id', 'serial_title', 'holdings_format', 'issns', 'currently_received', 'retention_policy',
                         'limited_retention_period', 'limited_retention_type', 'embargo_period',
                         'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']
    stored_values = {col: None for col in columns_to_update}


    for index, row in df_d.iterrows():
        if pd.notna(row['nlm_unique_id']):

            stored_values = {col: row[col] for col in columns_to_update}


        for col in columns_to_update:
            df_d.at[index, col] = stored_values[col]

    return df_d


def prepare(alma_nlm_merge_df, docline_df, print_or_electronic_choice):
    alma_nlm_merge_df['embargo_period'] = alma_nlm_merge_df['embargo_period'].fillna(0)
    alma_nlm_merge_df['nlm_unique_id'] = alma_nlm_merge_df['nlm_unique_id'].astype(str)





    alma_nlm_merge_df = alma_nlm_merge_df[(~alma_nlm_merge_df.isnull()['nlm_unique_id'] & alma_nlm_merge_df['record_type'].isin(['HOLDING'])) | ((alma_nlm_merge_df['record_type'].isin(['RANGE'])) & (~alma_nlm_merge_df.isnull()['begin_year']))].reset_index(drop=True)

    # alma_nlm_merge_df.loc[:,[begin_year', 'end_year']] = alma_nlm_merge_df.loc[:,[begin_year', 'end_year']].astype('float')
    # alma_nlm_merge_df.loc[:,[begin_year', 'end_year']] = alma_nlm_merge_df.loc[:,[begin_year', 'end_year']].astype('Int64')
    #

    # note so that the docline data can be treated as a real, valid table
    # data from the HOLDING line is copied over on to the RANGE line,
    # a relationship that in Docline format is only represented by the RANGE data
    # immediately following the HOLDING line.
    # this way they can be sorted

    # files_docline = glob.glob('Docline/*', recursive = True)

    # docline_filename = files_docline[0]
    # # Load the Docline CSV file
    # docline_df = pd.read_csv(docline_filename, dtype={'begin_year': 'Int64', 'end_year': 'Int64', 'begin_volume': 'Int64', 'end_volume': 'Int64', 'nlm_unique_id': 'str'}, engine='python')


    # Function to apply values in RANGE rows associated with holdings
    # simply by order, with al data from the holding row
    #
    # this was largely created by ChatGPT 4 with some tweaking

    # Apply the function to the DataFrame
    existing_docline_df = propagate_nlm_unique_id_and_libid_values(docline_df.copy())

    if print_or_electronic_choice == "1":
        alma_nlm_merge_df = alma_nlm_merge_df[alma_nlm_merge_df['holdings_format'] == 'Print']
        existing_docline_df = existing_docline_df[existing_docline_df['holdings_format'] == 'Print']
    elif print_or_electronic_choice == "2":
        alma_nlm_merge_df = alma_nlm_merge_df[alma_nlm_merge_df['holdings_format'] == 'Electronic']
        existing_docline_df = existing_docline_df[existing_docline_df['holdings_format'] == 'Electronic']
    # Convert 'embargo_period' and 'limited_retention_period' to integers
    existing_docline_df['embargo_period'] = existing_docline_df['embargo_period'].fillna(0).astype(int)
    existing_docline_df['limited_retention_period'] = existing_docline_df['limited_retention_period'].fillna(0).astype(int)

    # Define path for the output CSV file
    output_file_path = 'Processing/Existing Docline Holdings.csv'

    # Save the updated DataFrame to a CSV file
    existing_docline_df.to_csv(output_file_path, index=False)


    return [alma_nlm_merge_df, existing_docline_df]
def removeNA(string):

    if pd.isna(string) == False:
        list = string.split("; ")

        if "<NA>" in list:
            list.remove("<NA>")

        string = "; ".join(list)

    return(string)

### This function updates the value of 'currently_received'
### in the HOLDING row, if ANY of the RANGE rows for that
### title are currently received, i.e. do not have an end date
###
def apply_currently_received(updated_df):
    updated_HOLDING_df = updated_df.copy()
    updated_HOLDING_df = updated_df[updated_df['record_type'] == "HOLDING"]
    updated_RANGE_df = updated_df.copy()
    updated_RANGE_df = updated_df[updated_df['record_type'] == "RANGE"]


    for index, row in updated_HOLDING_df.iterrows():
        nlm_left = row['nlm_unique_id']
        holdings_format_left = row['holdings_format']
        action_left = row['action']
        rows_df = updated_RANGE_df[(updated_RANGE_df['nlm_unique_id'] == nlm_left) & (updated_RANGE_df['holdings_format'] == holdings_format_left)]
        currently_received_list = rows_df['end_year'].tolist()
        assignment_value = "No"
        for c_r in currently_received_list:

            if c_r == np.nan or c_r == 10000 or c_r == "10000" or c_r == "" or pd.isna(c_r):

                assignment_value = "Yes"
                break

        updated_df.loc[(updated_df['nlm_unique_id'] == nlm_left) & (updated_df['holdings_format'] == holdings_format_left) & (updated_df['record_type'] == "HOLDING"), 'currently_received'] = assignment_value

    return updated_df

####    This is a somewhat complex function that converts
####    what would otherwise be separate RANGE rows
####    corresponding to each portfolio for the journal in Alma
####    to a "compressed" list of RANGES such that none of them
####    overlap.
####
####    It also updates embargo values so that if there is any
####    holding that is not embargoed in the most recent years,
####    the journal is shown to be not embargoed

####
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def merge_intervals_optimized(df):
    """
    Merge overlapping or adjacent 'RANGE' holdings rows in a DataFrame,
    accounting for embargo periods and grouping by (nlm_unique_id, holdings_format).

    'HOLDING' rows are passed through unchanged.

    Logic:
    - A RANGE with end_year == 10000 is treated as open-ended ("infinite").
    - An open-ended range only extends the current merged range if it overlaps
      or directly abuts (no gap).
    - Embargo periods are respected when determining effective coverage.
    """

    # -------------------------------------------------------------------------
    # 1. Normalize and sort
    # -------------------------------------------------------------------------
    df = df.sort_values(
        by=['nlm_unique_id', 'holdings_format', 'record_type',
            'begin_year', 'end_year', 'embargo_period'],
        ascending=[True, True, True, True, True, True]
    ).copy()

    # Treat missing end_year as open-ended
    df.loc[df['record_type'] == 'RANGE', 'end_year'] = (
        df.loc[df['record_type'] == 'RANGE', 'end_year'].fillna(10000)
    )

    output_rows = []
    current_row = None

    def effective_end(row):
        embargo = row['embargo_period'] or 0
        return row['end_year'] - (embargo / 12.0)

    # -------------------------------------------------------------------------
    # 2. Merge iteration
    # -------------------------------------------------------------------------
    for _, row in df.iterrows():

        # HOLDING rows pass through
        if row['record_type'] == 'HOLDING':
            output_rows.append(row)
            continue

        if current_row is None:
            current_row = row
            continue

        # Different journal/format → flush current_row
        if (row['nlm_unique_id'] != current_row['nlm_unique_id'] or
            row['holdings_format'] != current_row['holdings_format']):
            output_rows.append(current_row)
            current_row = row
            continue

        left_eff = effective_end(row)
        right_eff = effective_end(current_row)

        # ---------------------------------------------------------------------
        # Case A: Non-overlapping (gap) → finalize current_row, start new one
        # ---------------------------------------------------------------------
        if row['begin_year'] > current_row['end_year']:
            output_rows.append(current_row)
            current_row = row
            continue

        # ---------------------------------------------------------------------
        # Case B: Overlapping OR touching
        # ---------------------------------------------------------------------
        # Infinity rule applies only if overlapping or adjacent
        # ---------------------------------------------------------------------
        overlapping = row['begin_year'] <= current_row['end_year'] + 1

        if overlapping and row['end_year'] == 10000:
            # Infinity extends the current range to 10000
            current_row['end_year'] = 10000
            current_row['embargo_period'] = row['embargo_period']
            continue

        # ---------------------------------------------------------------------
        # Case C: Overlapping, same embargo → extend
        # ---------------------------------------------------------------------
        if overlapping and row['embargo_period'] == current_row['embargo_period']:
            current_row['end_year'] = max(current_row['end_year'], row['end_year'])
            continue

        # ---------------------------------------------------------------------
        # Case D: Overlapping, different embargo → prefer broader effective coverage
        # ---------------------------------------------------------------------
        if overlapping:
            if left_eff > right_eff:
                current_row['end_year'] = row['end_year']
                current_row['embargo_period'] = row['embargo_period']
            elif np.isclose(left_eff, right_eff):
                if row['embargo_period'] < current_row['embargo_period']:
                    current_row['end_year'] = row['end_year']
                    current_row['embargo_period'] = row['embargo_period']
            continue

        # ---------------------------------------------------------------------
        # Otherwise, they’re truly disjoint (shouldn't merge)
        # ---------------------------------------------------------------------
        output_rows.append(current_row)
        current_row = row

    # -------------------------------------------------------------------------
    # 3. Append last range
    # -------------------------------------------------------------------------
    if current_row is not None:
        output_rows.append(current_row)

    # -------------------------------------------------------------------------
    # 4. Return merged DataFrame
    # -------------------------------------------------------------------------
    output_df = pd.DataFrame(output_rows, columns=df.columns)
    return output_df



def merge(alma_nlm_merge_df, existing_docline_df):
    #####################################################
    #####################################################
    ####    This is the final merge: existing Docline
    ####    against the dataframe of parsed Alma holdings compared
    ####    against NLM data
    ####
    ####    Method:
    ####        - match by nlm_unique_id, substracting
    ####          the "NLM_" prefix from the Docline data for the purpose of the match
    ####        - before the merge, we need to separate
    ####          "Deleted" and "ILL Not Allowed" from In "Repository" into separate dataframes
    ####          ILL Not Allowed is parsed from a field named "ILL Allowed", and comes from Analytics
    ####          You Analytics report should populate this field according to your local practices.
    ####          But you can refer to __ for a guide on how Tufts records and parses this
    ####          /shared/Community/Reports/Institutions/Tufts University/	Electronic Journals - for NLM-Docline-ILL Not Allowed
    ####          In our instance, we process this load separately, of records to remove from Docline for the above reasons
    ####          The deleted rows will be added to the final
    ####          output dataframe with simply the DELETE action,
    ####          if they match
    ####        - before merging sort each dataframe according to the following column sort orders
    ####            - alphabetically by title THEN
    ####            - alphabetically by NLM Unique ID THEN
    ####            - numerically by start year THEN
    ####            - numerically by start volume THEN
    ####            - numerically by end year THEN
    ####            - numerically by end volume
    ####

    #######
    # Generally to avoid having changes affect other dataframes,
    # the script makes copies of an existing DataFrame
    # to get the column format, and then makes chages to the copy
    # This is a property of pandas that unless you specifically specify
    # a copy even actions on dataframes that are not being assigned to, that
    # dataframe will result in mutations to the base dataframe


    existing_docline_df['nlm_unique_id'] = existing_docline_df['nlm_unique_id'].astype(str)
    alma_nlm_merge_df['nlm_unique_id'] = alma_nlm_merge_df['nlm_unique_id'].astype(str)
    alma_nlm_merge_df['begin_volume'] = ""
    alma_nlm_merge_df['end_volume'] = ""
    deleted_alma_df = pd.DataFrame(columns=alma_nlm_merge_df.columns)
    #
    if 'Bibliographic Lifecycle' in deleted_alma_df and 'ILL Allowed' in deleted_alma_df:

        deleted_alma_df = deleted_alma_df.loc[(deleted_alma_df['Bibliographic Lifecycle'] == 'Deleted') | (deleted_alma_df['ILL Allowed'] == 'ILL Not Allowed')]

    elif 'Bibliographic Lifecycle' in deleted_alma_df and 'ILL Allowed' not in deleted_alma_df:
        deleted_alma_df = deleted_alma_df.loc[deleted_alma_df['Bibliographic Lifecycle'] == 'Deleted']

    elif 'Lifecycle' in deleted_alma_df and 'ILL Allowed' in deleted_alma_df:
        deleted_alma_df = deleted_alma_df.loc[(deleted_alma_df['Lifecycle'] == 'Deleted') | (deleted_alma_df['ILL Allowed'] == 'ILL Not Allowed')]
    elif 'Lifecycle' in deleted_alma_df and 'ILL Allowed' not in deleted_alma_df:
        deleted_alma_df = deleted_alma_df.loc[(deleted_alma_df['Lifecycle'] == 'Deleted')]
    elif 'ILL Allowed' in deleted_alma_df and 'Lifecycle' not in deleted_alma_df and 'Bibliographic Lifecycle' not in deleted_alma_df:
        deleted_alma_df = deleted_alma_df.loc[deleted_alma_df['ILL Allowed'] == 'ILL Not Allowed']


    current_alma_df = alma_nlm_merge_df.copy()

    current_alma_df = current_alma_df[~current_alma_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(deleted_alma_df.set_index(['nlm_unique_id', 'holdings_format']).index)]

    deleted_alma_df = deleted_alma_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'begin_volume', 'end_year', 'end_volume'], ascending = [True, True, True, True, True, True], na_position = 'first')

    current_alma_df = current_alma_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'begin_volume', 'end_year', 'end_volume'], ascending = [True, True, True, True, True, True], na_position = 'first')


    existing_docline_df_for_compare = existing_docline_df.copy()
    existing_docline_df_for_compare['nlm_unique_id'] = existing_docline_df_for_compare['nlm_unique_id'].apply(lambda x: x.replace("NLM_", ""))


    existing_docline_df_for_compare['end_year'] = existing_docline_df_for_compare['end_year'].apply(lambda x: str(x).replace("10000", ""))
    existing_docline_df_for_compare['end_year'] = pd.to_numeric(existing_docline_df_for_compare['end_year'], errors='coerce')
    existing_docline_df_for_compare['end_year'] = existing_docline_df_for_compare['end_year'].astype('Int64')
    existing_docline_df_for_compare['end_year'] = existing_docline_df_for_compare['end_year'].replace(0, np.nan)

    existing_docline_df_for_compare['begin_year'] = existing_docline_df_for_compare['begin_year'].apply(lambda x: str(x).replace("10000", ""))
    existing_docline_df_for_compare['begin_year'] = pd.to_numeric(existing_docline_df_for_compare['begin_year'], errors='coerce')
    existing_docline_df_for_compare['begin_year'] = existing_docline_df_for_compare['begin_year'].astype('Int64')
    existing_docline_df_for_compare['begin_year'] = existing_docline_df_for_compare['begin_year'].replace(0, np.nan)

    existing_docline_df_for_compare['end_volume'] = existing_docline_df_for_compare['end_volume'].apply(lambda x: str(x).replace("10000", ""))
    existing_docline_df_for_compare['end_volume'] = pd.to_numeric(existing_docline_df_for_compare['end_volume'], errors='coerce')
    existing_docline_df_for_compare['end_volume'] = existing_docline_df_for_compare['end_volume'].astype('Int64')
    existing_docline_df_for_compare['end_volume'] = existing_docline_df_for_compare['end_volume'].replace(0, np.nan)

    existing_docline_df_for_compare['begin_volume'] = existing_docline_df_for_compare['begin_volume'].apply(lambda x: str(x).replace("10000", ""))
    existing_docline_df_for_compare['begin_volume'] = pd.to_numeric(existing_docline_df_for_compare['begin_volume'], errors='coerce')
    existing_docline_df_for_compare['begin_volume'] = existing_docline_df_for_compare['begin_volume'].astype('Int64')
    existing_docline_df_for_compare['begin_volume'] = existing_docline_df_for_compare['begin_volume'].replace(0, np.nan)


    deleted_output_df = deleted_alma_df.copy()

    deleted_output_df = deleted_output_df[deleted_output_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(existing_docline_df_for_compare.set_index(['nlm_unique_id', 'holdings_format']).index)]

    deleted_output_df.to_excel("Processing/Deleted Output DF.xlsx", index=False)
    existing_docline_df_for_compare.to_excel('Processing/Existing Docline for Compare.xlsx', index=False)



    current_alma_df['end_year'] = pd.to_numeric(current_alma_df['end_year'], errors='coerce')
    current_alma_df['end_year'] = current_alma_df['end_year'].astype('Int64')
    current_alma_df['end_year'] = current_alma_df['end_year'].replace(0, np.nan)



    current_alma_df['begin_year'] = pd.to_numeric(current_alma_df['begin_year'], errors='coerce')
    current_alma_df['begin_year'] = current_alma_df['begin_year'].astype('Int64')
    current_alma_df['begin_year'] = current_alma_df['begin_year'].replace(0, np.nan)


    current_alma_df['embargo_period'] = current_alma_df['embargo_period'].apply(lambda x: str(x).replace("10000", ""))
    current_alma_df['embargo_period'] = pd.to_numeric(current_alma_df['embargo_period'], errors='coerce')
    current_alma_df['embargo_period'] = current_alma_df['embargo_period'].astype('Int64')

        # ------------------------------------------------------------
    # HARD VALIDATION: every RANGE row must have a begin_year
    # If missing, send to error output and remove from processing
    # ------------------------------------------------------------
    error_missing_begin_year_df = current_alma_df[
        (current_alma_df["record_type"] == "RANGE") & (current_alma_df["begin_year"].isna())
    ].copy()

    if not error_missing_begin_year_df.empty:
        error_missing_begin_year_df["Error Message"] = (
            "Could not read begin date of portfolio from coverage statement"
        )
        # remove these rows so merge_intervals_optimized won't choke on <NA>
        current_alma_df = current_alma_df.drop(index=error_missing_begin_year_df.index)
    else:
        # keep it defined for the final output write pass
        error_missing_begin_year_df = pd.DataFrame(columns=list(current_alma_df.columns) + ["Error Message"])

    current_alma_df.to_csv('Processing/compressed_before_optimization.csv', index=False)

    current_alma_compressed_df = merge_intervals_optimized(current_alma_df.copy())

    current_alma_compressed_df.to_csv('Processing/compressed_after_optimization.csv', index=False)

    current_alma_compressed_df.sort_values(by=['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year'], inplace=True)

    current_alma_compressed_df.loc[(current_alma_compressed_df['end_year'] == 10000) & (current_alma_compressed_df['record_type'] == "RANGE"), 'currently_received'] = "Yes"

    current_alma_compressed_df['end_year'] = current_alma_compressed_df['end_year'].replace(to_replace=10000, value=np.nan)


    current_alma_compressed_df = apply_currently_received(current_alma_compressed_df)

    current_alma_compressed_df.to_csv('Analysis/alma_df_after_currently_received_handling.csv', index=False)



    # cases where the NLM unique ID and holdings format combination
    # between the Alma data
    # (with the NLM Unique ID coming from the NLM catalog as noted above)
    # match are more complicated
    # first, there are those records that match exactly what is in Docline,
    # including RANGE.  These can be discluded from the import as they remain the same

    # things that match on NLM Unique ID and holdings format,
    # but differ in either range or other holdings field need to be updated
    # This requires that the script first submit a DELETE command for what
    # you have in Docline now,
    # and then ADD for the new holding and Ranges,
    # basically a drawn out UPDATE, which doesn't exist in Docline processing itself

    # the way this is accomplished is, once the two tables are each sorted
    # with the 6-tier sort noted above,
    # rolling up the range rows into the holding rows with groupby
    # for each table (keep in mind for the purpose of this analysis, the holding data is repeated on range lines),
    # again with sorted range data,
    # the entire row of the Alma DataFrame is compared against the entire row of the
    # Docline table, minus a few fields like updated date, with .isin() and setting the entire column list as the
    # match "field" or in pandas , a multi-colmn index.
    # Again full matches like these do not need to be included because they do not change.
    # For those that NLM Unique ID and holdings format match,
    # that don't match for the rest of the fields (other holding data, and/or aggregated range data)
    # the script issues the Docline data with a DELETE action
    # and then immediately after add the Alma data as an ADD



    current_alma_compressed_df['end_year'] = current_alma_compressed_df['end_year'].apply(lambda x: str(x).replace("10000", ""))
    current_alma_compressed_df['begin_year'] = current_alma_compressed_df['begin_year'].replace("", '0')
    current_alma_compressed_df['begin_year'] = current_alma_compressed_df['begin_year'].replace("<NA>", '0')

    current_alma_compressed_df['end_year'] = current_alma_compressed_df['end_year'].replace("", '0')
    current_alma_compressed_df['end_year'] = current_alma_compressed_df['end_year'].replace("<NA>", '0')


    current_alma_compressed_df['begin_year'] = current_alma_compressed_df['begin_year'].replace('0', np.nan)
    current_alma_compressed_df['end_year'] = current_alma_compressed_df['end_year'].replace('0', np.nan)
    current_alma_compressed_df['begin_year'] = current_alma_compressed_df['begin_year'].replace(0, np.nan)
    current_alma_compressed_df['end_year'] = current_alma_compressed_df['end_year'].replace(0, np.nan)
    current_alma_compressed_df['embargo_period'] = current_alma_compressed_df['embargo_period'].replace("", 0)





    current_alma_compressed_df['begin_year'] = current_alma_compressed_df['begin_year'].astype('float', errors='ignore')
    current_alma_compressed_df['begin_year'] = current_alma_compressed_df['begin_year'].astype('Int64', errors='ignore')
    current_alma_compressed_df['end_year'] = current_alma_compressed_df['end_year'].astype('float', errors='ignore')

    current_alma_compressed_df['end_year'] = current_alma_compressed_df['end_year'].astype('Int64', errors='ignore')
    current_alma_compressed_df['embargo_period'] = current_alma_compressed_df['embargo_period'].astype('float', errors='ignore')
    current_alma_compressed_df['embargo_period'] = current_alma_compressed_df['embargo_period'].astype('Int64', errors='ignore')

    current_alma_compressed_df = current_alma_compressed_df.reset_index()
    existing_docline_df_for_compare = existing_docline_df_for_compare.reset_index()
    current_alma_compressed_df = current_alma_compressed_df.drop('Bibliographic Lifecycle', axis=1)


    try:
        current_alma_compressed_df = current_alma_compressed_df.drop('index', axis=1)
    except:
        drop_index = False
    existing_docline_df_for_compare = existing_docline_df_for_compare.drop('index', axis=1)
    matched_base_df_alma_side = current_alma_compressed_df[current_alma_compressed_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(existing_docline_df_for_compare.set_index(['nlm_unique_id', 'holdings_format']).index)]
    matched_base_df_docline_side = existing_docline_df_for_compare[existing_docline_df_for_compare.set_index(['nlm_unique_id', 'holdings_format']).index.isin(current_alma_df.set_index(['nlm_unique_id', 'holdings_format']).index)]
    pd.set_option('display.max_columns', None)

    matched_base_df_alma_side['end_year'] = matched_base_df_alma_side['end_year'].astype('Int64')
    matched_base_df_alma_side['end_year'] = matched_base_df_alma_side['end_year'].apply(lambda x: str(x).replace("10000", ""))
    matched_base_df_alma_side['end_year'] = matched_base_df_alma_side['end_year'].apply(lambda x: x.replace("<NA>", ""))
    matched_base_df_alma_side['end_year'] = matched_base_df_alma_side['end_year'].fillna("")

    matched_base_df_docline_side['embargo_period'] = matched_base_df_docline_side['embargo_period'].astype('Int64')
    matched_base_df_docline_side['embargo_period'] = matched_base_df_docline_side['embargo_period'].fillna(0)
    matched_base_df_docline_side[matched_base_df_docline_side['embargo_period'] == ""] = 0


    # These are the Alma records to add to Docline, because they are not
    # in current Docline holdings by NLM Unique ID
    # Use compressed ranges for "totally new" titles too
    add_df = current_alma_compressed_df[
      ~current_alma_compressed_df.set_index(['nlm_unique_id', 'holdings_format']).index
      .isin(existing_docline_df_for_compare.set_index(['nlm_unique_id', 'holdings_format']).index)
    ]


    # These are the Docline records that did not match to NLM-enriched Alma
    # data by NLM unique ID.  They should be reviewed manually
    in_docline_only_preserve_df = existing_docline_df_for_compare.copy()
    in_docline_only_preserve_df = existing_docline_df_for_compare[~existing_docline_df_for_compare.set_index(['nlm_unique_id', 'holdings_format']).index.isin(alma_nlm_merge_df.set_index(['nlm_unique_id', 'holdings_format']).index)]


    all_columns = existing_docline_df_for_compare.columns


    aggregate_columns = []
    group_by_columns = []
    all_columns = all_columns.tolist()

    all_columns.remove('last_modified')
    all_columns.remove('issns')

    all_columns.remove('begin_volume')
    all_columns.remove('end_volume')
    all_columns.remove('action')
    all_columns.remove('ignore_warnings')
    try:
        all_columns.remove('libid')


    except:
        libid = ""
    all_columns.remove('serial_title')
    for column in all_columns:
        if column in ['begin_year', 'end_year', 'embargo_period']:
            aggregate_columns.append(column)
        elif column == 'Bibliographic Lifecycle' or column == 'last_modified' or column == 'issns' or column == 'begin_volume' or column == 'end_volume':
            continue
        else:
            group_by_columns.append(column)







    # in this section, once the range columns and rows of
    # both the Alma-NLM data set and the existing Docline set
    # have been sorted in 6-tiered sort noted above,
    # all the range rows are rolled up into the holding
    # rows such that each title should appear only once
    #
    # With this in place, the whole rows (minus modified date and a few other fields)
    # from the Docline-formatted Alma set can be compared with whole rows from
    # the current Docline set to determine if the holdings *and* ranges are a complete match,
    # in which case nothing needs to be done with them,
    # or if any holding or range data have been added or changed.
    # in  which case a DELETE action is needed for the current Docline
    # record and an ADD action for the incoming Alma holding and ranges, even
    # though they match by NLM Unique ID, due to requirements listed
    # in the Docline user manual.

    # These dataframes are only used to filter the final output dataframes
    # because in these comparison dataframes
    # the holdings and ranges are on the same line, which
    # is not the Docline format




    # dealing with date and volume columns that are float because they
    # contain numbers and NaN
    # https://stackoverflow.com/questions/60024262/error-converting-object-string-to-int32-typeerror-object-cannot-be-converted


    matched_base_df_alma_side['begin_year'] = matched_base_df_alma_side['begin_year'].replace("", '0')
    matched_base_df_alma_side['begin_year'] = matched_base_df_alma_side['begin_year'].replace("<NA>", '0')

    matched_base_df_alma_side['end_year'] = matched_base_df_alma_side['end_year'].replace("", '0')
    matched_base_df_alma_side['end_year'] = matched_base_df_alma_side['end_year'].replace("<NA>", '0')


    matched_base_df_alma_side['begin_year'] = matched_base_df_alma_side['begin_year'].replace('0', np.nan)
    matched_base_df_alma_side['end_year'] = matched_base_df_alma_side['end_year'].replace('0', np.nan)
    matched_base_df_alma_side['begin_year'] = matched_base_df_alma_side['begin_year'].replace(0, np.nan)
    matched_base_df_alma_side['end_year'] = matched_base_df_alma_side['end_year'].replace(0, np.nan)
    matched_base_df_alma_side['embargo_period'] = matched_base_df_alma_side['embargo_period'].replace("", 0)





    matched_base_df_alma_side['begin_year'] = matched_base_df_alma_side['begin_year'].astype('float', errors='ignore')
    matched_base_df_alma_side['begin_year'] = matched_base_df_alma_side['begin_year'].astype('Int64', errors='ignore')
    matched_base_df_alma_side['end_year'] = matched_base_df_alma_side['end_year'].astype('float', errors='ignore')

    matched_base_df_alma_side['end_year'] = matched_base_df_alma_side['end_year'].astype('Int64', errors='ignore')
    matched_base_df_alma_side['embargo_period'] = matched_base_df_alma_side['embargo_period'].astype('float', errors='ignore')
    matched_base_df_alma_side['embargo_period'] = matched_base_df_alma_side['embargo_period'].astype('Int64', errors='ignore')

    existing_docline_df_for_compare['begin_year'] = existing_docline_df_for_compare['begin_year'].replace(0, np.nan)
    existing_docline_df_for_compare['embargo_period'] = existing_docline_df_for_compare['embargo_period'].replace("", np.nan)


    existing_docline_df_for_compare['begin_year'] = existing_docline_df_for_compare['begin_year'].replace("", '0')
    existing_docline_df_for_compare['begin_year'] = existing_docline_df_for_compare['begin_year'].replace("<NA>", '0')

    existing_docline_df_for_compare['end_year'] = existing_docline_df_for_compare['end_year'].replace("", '0')
    existing_docline_df_for_compare['end_year'] = existing_docline_df_for_compare['end_year'].replace("<NA>", '0')


    existing_docline_df_for_compare['begin_year'] = existing_docline_df_for_compare['begin_year'].replace('0', np.nan)
    existing_docline_df_for_compare['end_year'] = existing_docline_df_for_compare['end_year'].replace(0, np.nan)





    existing_docline_df_for_compare['begin_year'] = existing_docline_df_for_compare['begin_year'].astype('float', errors='ignore')
    existing_docline_df_for_compare['begin_year'] = existing_docline_df_for_compare['begin_year'].astype('Int64', errors='ignore')
    existing_docline_df_for_compare['end_year'] = existing_docline_df_for_compare['end_year'].astype('float', errors='ignore')
    existing_docline_df_for_compare['end_year'] = existing_docline_df_for_compare['end_year'].astype('Int64', errors='ignore')
    existing_docline_df_for_compare['embargo_period'] = existing_docline_df_for_compare['embargo_period'].astype('float', errors='ignore')
    existing_docline_df_for_compare['embargo_period'] = existing_docline_df_for_compare['embargo_period'].astype('Int64', errors='ignore')



    # note the group by columns being those fields in the holding row,
    # and the aggregate columns being those things in range, that are being rolled up
    existing_docline_for_compare_agg_df = existing_docline_df_for_compare.copy()
    matched_nlm_alma_df_for_compare_agg = matched_base_df_alma_side.copy()



    matched_nlm_alma_df_for_compare_agg = matched_nlm_alma_df_for_compare_agg.sort_values(by = ['nlm_unique_id', 'holdings_format', 'record_type', 'begin_year', 'end_year', 'embargo_period'], ascending = [True, True, True, True, True, True], na_position = 'first')
    existing_docline_for_compare_agg_df = existing_docline_for_compare_agg_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'record_type', 'begin_year', 'end_year', 'embargo_period'], ascending = [True, True, True, True, True, True], na_position = 'first')



    existing_docline_for_compare_agg_df = existing_docline_for_compare_agg_df.groupby(group_by_columns, dropna=False, as_index=False).agg({aggregate_columns[0]: lambda x: '; '.join(set(x.astype(str))), aggregate_columns[1]: lambda x: '; '.join(set(x.astype(str))), aggregate_columns[2]: lambda x: '; '.join(set(x.astype(str)))})
    matched_nlm_alma_df_for_compare_agg = matched_nlm_alma_df_for_compare_agg.groupby(group_by_columns, dropna=False, as_index=False).agg({aggregate_columns[0]: lambda x: '; '.join(set(x.astype(str))), aggregate_columns[1]: lambda x: '; '.join(set(x.astype(str))), aggregate_columns[2]: lambda x: '; '.join(set(x.astype(str)))})


    # in the rolled up range data, remove the appearance of "<NA>" in the stringified list

    full_merge_on_just_nlm_id_and_format = pd.merge(existing_docline_for_compare_agg_df, matched_nlm_alma_df_for_compare_agg, how='inner', on=['nlm_unique_id', 'holdings_format'])


    full_matched_for_compare_agg_df = matched_nlm_alma_df_for_compare_agg.copy()
    full_matched_for_compare_agg_df = full_matched_for_compare_agg_df.reset_index()




    existing_docline_for_compare_agg_df = existing_docline_for_compare_agg_df.reset_index()

    # we don't need to include the nlm unique id or format criterion here
    # explicitly because all_columns includes this

    full_matched_for_compare_agg_df = full_matched_for_compare_agg_df[full_matched_for_compare_agg_df.set_index(all_columns).index.isin(existing_docline_for_compare_agg_df.set_index(all_columns).index)]

    # These are the Alma records in which the NLM Unique ID matches,
    # but the entire aggregated row (from above) doesn't match,
    # i.e either the holding data or the range data is different.
    # if the holding data is different, it's possible that you've made some
    # different decisions about fields such as limited_retention_policy than you have in the past

    # as mentioned above, this is aggregated so it won't *be* the output
    # data but will be used to filter it


    matched_nlm_alma_df_for_compare_agg = matched_nlm_alma_df_for_compare_agg.reset_index()
    different_ranges_alma_compare_df_agg = matched_nlm_alma_df_for_compare_agg.copy()

    try:
        different_ranges_alma_compare_df_agg = different_ranges_alma_compare_df_agg.drop('index', axis=1)

    except:
        value = False
    try:
        different_ranges_alma_compare_df_agg = different_ranges_alma_compare_df_agg.drop('libid', axis=1)

    except:
        value = False


    try:
        existing_docline_for_compare_agg_df = existing_docline_for_compare_agg_df.drop('index', axis=1)

    except:
        value = False
    try:
        existing_docline_for_compare_agg_df = existing_docline_for_compare_agg_df.drop('libid', axis=1)

    except:
        value = False



    try:
        different_ranges_docline_compare_df_agg = different_ranges_docline_compare_df_agg.drop('index', axis=1)

    except:
        value = False

    different_ranges_alma_compare_df_agg = different_ranges_alma_compare_df_agg[(different_ranges_alma_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index.isin(existing_docline_for_compare_agg_df.set_index(['nlm_unique_id', 'holdings_format']).index)) & (~different_ranges_alma_compare_df_agg.set_index(all_columns).index.isin(existing_docline_for_compare_agg_df.set_index(all_columns).index))]

    # ditto to above, but among the current Docline set compared to Alma (other direction)
    different_ranges_docline_compare_df_agg = existing_docline_for_compare_agg_df.copy()

    different_ranges_docline_compare_df_agg = different_ranges_docline_compare_df_agg[(different_ranges_docline_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index.isin(different_ranges_alma_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index))]



    # now take the output produced with the iteration loops, and test it for
    # membership in the above dataframes processed for testing,
    # to get the final output dataframes
    full_match_output_df = alma_nlm_merge_df.copy()
    full_match_output_df = full_match_output_df.reset_index()
    full_match_output_df = full_match_output_df[full_match_output_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(full_matched_for_compare_agg_df.set_index(['nlm_unique_id', 'holdings_format']).index)]

    counts_df = pd.DataFrame(columns=['Set', 'Number of Rows', 'Number of NLM Unique IDs'])

    counts_df = pd.concat([counts_df, pd.DataFrame({'Set': 'Full Match', 'Number of Rows':len(full_match_output_df), 'Number of NLM Unique IDs': len(pd.unique(full_match_output_df['nlm_unique_id']))}, index=[0])])



    different_ranges_alma_output_df = current_alma_compressed_df.copy()
    different_ranges_alma_output_df = different_ranges_alma_output_df[different_ranges_alma_output_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(different_ranges_alma_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index)]
    different_ranges_alma_output_df = different_ranges_alma_output_df.reset_index()
    counts_df = pd.concat([counts_df, pd.DataFrame({'Set': 'Alma Different Ranges', 'Number of Rows': len(different_ranges_alma_output_df), 'Number of NLM Unique IDs': len(pd.unique(different_ranges_alma_output_df['nlm_unique_id']))}, index=[0])])
    different_ranges_alma_output_df = different_ranges_alma_output_df.reset_index()

    different_ranges_docline_output_df = existing_docline_df_for_compare.copy()
    different_ranges_docline_output_df = existing_docline_df_for_compare[existing_docline_df_for_compare.set_index(['nlm_unique_id', 'holdings_format']).index.isin(different_ranges_docline_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index)]

    counts_df = pd.concat([counts_df, pd.DataFrame({'Set': 'Docline Different Ranges', 'Number of Rows': len(different_ranges_docline_output_df), 'Number of NLM Unique IDs': len(pd.unique(different_ranges_docline_output_df['nlm_unique_id']))}, index=[0])])

    add_df['nlm_unique_id'] = add_df['nlm_unique_id'].apply(lambda x: "NLM_" + x)
    deleted_output_df['nlm_unique_id'] = deleted_output_df['nlm_unique_id'].apply(lambda x: "NLM_" + x)

    full_match_output_df['nlm_unique_id'] = full_match_output_df['nlm_unique_id'].apply(lambda x: "NLM_" + x)
    different_ranges_alma_output_df['nlm_unique_id'] = different_ranges_alma_output_df['nlm_unique_id'].apply(lambda x: "NLM_" + x)
    different_ranges_docline_output_df['nlm_unique_id'] = different_ranges_docline_output_df['nlm_unique_id'].apply(lambda x: "NLM_" + x)

    #do I need this


    add_df['action'] = 'ADD'
    deleted_output_df['action'] = 'DELETE'

    full_match_output_df['action'] = 'ADD'
    different_ranges_alma_output_df['action'] = 'ADD'
    different_ranges_docline_output_df['action']= 'DELETE'

    different_ranges_alma_output_df = different_ranges_alma_output_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True], na_position = 'first')
    different_ranges_docline_output_df = different_ranges_docline_output_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True], na_position = 'first')

    no_dates_df = different_ranges_alma_output_df.copy()

    no_dates_df = different_ranges_alma_output_df[different_ranges_alma_output_df['begin_year'].isna()]

    # NOTE: defer writing No Dates output until final output section
    different_ranges_alma_output_df = different_ranges_alma_output_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True, True], na_position = 'first')

    try:
        different_ranges_alma_output_df = different_ranges_alma_output_df.drop('index_0', axis=1)
    except:
        value = False

    try:
        different_ranges_alma_output_df = different_ranges_alma_output_df.drop('index', axis=1)
    except:
        value = False


    different_ranges_alma_output_df.to_csv('Processing/Different Ranges Alma Before Compression.csv', index=False)
    try:
        different_ranges_alma_output_df = different_ranges_alma_output_df.drop(columns=['libid', 'Lifecycle'])
    except:
        drop = False
    merged_updated_df = pd.concat([different_ranges_alma_output_df, different_ranges_docline_output_df])

    merged_updated_df = merged_updated_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True, True], na_position = 'first')

    merged_updated_df['ignore_warnings'] = "Yes"
    merged_updated_df = merged_updated_df[((~merged_updated_df['begin_year'].isna()) | (~merged_updated_df['begin_volume'].isna()) | (merged_updated_df['record_type'] == "HOLDING"))]

    no_dates_add_df = add_df.copy()


    no_dates_add_df = add_df[add_df['begin_year'].isna()]


    add_df = add_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True, True], na_position = 'first')



    add_df = add_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True, True], na_position = 'first')


    try:
        different_ranges_docline_output_df = different_ranges_docline_output_df.drop('index', axis=1)

    except:
        print('index already removed')

    try:
        different_ranges_alma_output_df = different_ranges_alma_output_df.drop('index', axis=1)

    except:
        print('index already removed')
    try:
        different_ranges_alma_output_df = different_ranges_alma_output_df.drop('level_0', axis=1)
    except:
        print("level 0 already removed")
    try:
        different_ranges_alma_output_df = different_ranges_alma_output_df.drop('level_0', axis=1)
    except:
        print("level 0 already removed")
    try:
        different_ranges_alma_output_df = different_ranges_alma_output_df.drop('Bibliographic Lifecycle', axis=1)

    except:
        "Bibliographic Lifecycle column already removed"





    counts_df = pd.concat([counts_df, pd.DataFrame({'Set': 'Alma Adds', 'Number of Rows': len(add_df), 'Number of NLM Unique IDs': len(pd.unique(add_df['nlm_unique_id']))}, index=[0])])


    add_df = add_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, True, True, True, True], na_position = 'first')

    deleted_output_df = deleted_output_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, True, True, True, True], na_position = 'first')

    full_match_output_df = full_match_output_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'embargo_period', 'record_type', 'begin_year', 'end_year'], ascending = [True, True, True, True, True, True])
    in_docline_only_preserve_df = in_docline_only_preserve_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'end_year'], ascending = [True, True, True, True], na_position = 'first')


    try:
        add_df = add_df.drop('index', axis=1)

    except:

        print('index already removed')
    try:
        deleted_output_df = deleted_output_df.drop('index', axis=1)
        print(nothing)
    except:
        print('index already removed')

    try:
        deleted_output_df = deleted_output_df.drop('Bibliographic Lifecycle', axis=1)

    except:
        "Bibliographic Lifecycle column already removed"
    try:
        full_match_output_df = full_match_output_df.drop('index', axis=1)

    except:
        print('index already removed')

    try:
        merged_updated_df = merged_updated_df.drop('Lifecycle', axis=1)

    except:
        print("Lifecycle column already removed")
    try:
        merged_updated_df = merged_updated_df.drop('level_0', axis=1)
    except:
        print("level 0 already removed")

    try:
        merged_updated_df = merged_updated_df.drop('Lifecycle', axis=1)

    except:
        print("Lifecycle column already removed")
    try:
        merged_updated_df = merged_updated_df.drop('Bibliographic Lifecycle', axis=1)

    except:
        print("Bibliographic Lifecycle column already removed")

    try:
        merged_updated_df = merged_updated_df.drop('libid', axis=1)

    except:
        print("libid column already removed")


    try:
        add_df = add_df.drop('level_0', axis=1)
    except:
        print("level 0 already removed")
    try:
        add_df = add_df.drop('index', axis=1)
    except:
        print("index already removed")
    try:
        add_df = add_df.drop('Lifecycle', axis=1)

    except:
        print("Lifecycle column already removed")

    try:
        add_df = add_df.drop('Bibliographic Lifecycle', axis=1)

    except:
        print("Bibliographic Lifecycle column already removed")

        try:
            merged_updated_df = merged_updated_df.drop('libid', axis=1)

        except:
            print("libid column already removed")

    
   


    try:
        merged_updated_df = merged_updated_df.drop('index', axis=1)

    except:
        "index column already removed"
    counts_df = pd.concat([counts_df, pd.DataFrame({'Set': 'Deleted from Alma', 'Number of Rows': len(deleted_output_df), 'Number of NLM Unique IDs': len(pd.unique(deleted_output_df['nlm_unique_id']))}, index=[0])])




    full_match_output_df.to_csv('Output/Full Match Final.csv', index=False)
    merged_updated_df.to_csv('Output/Update Final.csv', index=False)
    #different_ranges_docline_output_df.to_csv('Output/Different Ranges Docline Final.csv', index=False)
    different_ranges_alma_output_df.to_csv('Output/Different Ranges Alma Final.csv', index=False)
    #



    merged_updated_df = pd.read_csv('Output/Update Final.csv', engine="python")#dtype={'begin_year': 'Int64', 'end_year': 'Int64', 'begin_volume': 'Int64', 'end_volume': 'Int64', 'nlm_unique_id': 'str'}, engine="python")
    full_match_output_df = pd.read_csv('Output/Full Match Final.csv', engine="python")#dtype={'begin_year': 'Int64', 'end_year': 'Int64', 'begin_volume': 'Int64', 'end_volume': 'Int64', 'nlm_unique_id': 'str'}, engine="python")
    #different_ranges_alma_output_df = pd.read_csv('Output/Different Ranges Alma Final.csv', engine="python")#dtype={'begin_year': 'Int64', 'end_year': 'Int64', 'begin_volume': 'Int64', 'end_volume': 'Int64', 'nlm_unique_id': 'str'}, engine="python")
    merged_updated_df[['begin_year', 'end_year', 'begin_volume', 'end_volume']] = merged_updated_df[['begin_year', 'end_year', 'begin_volume', 'end_volume']].astype('Int64')

    #merged_updated_df.loc[(merged_updated_df['record_type'] == 'RANGE') & (merged_updated_df['action'] == 'ADD'), ['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_period', 'limited_retention_type', 'embargo_period', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']] = ""
    cols_to_convert = ['begin_year', 'end_year', 'begin_volume', 'end_volume', 'embargo_period', 'limited_retention_period']

    merged_updated_df[cols_to_convert] = merged_updated_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    merged_updated_df[cols_to_convert] = merged_updated_df[cols_to_convert].astype("Int64")
    merged_updated_df[cols_to_convert] = merged_updated_df[cols_to_convert].replace(0, np.nan)

    full_match_output_df[cols_to_convert] = full_match_output_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    full_match_output_df[cols_to_convert] = full_match_output_df[cols_to_convert].astype("Int64")
    full_match_output_df[cols_to_convert] = full_match_output_df[cols_to_convert].replace(0, np.nan)

    different_ranges_alma_output_df[cols_to_convert] = different_ranges_alma_output_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    different_ranges_alma_output_df[cols_to_convert] = different_ranges_alma_output_df[cols_to_convert].astype("Int64")
    different_ranges_alma_output_df[cols_to_convert] = different_ranges_alma_output_df[cols_to_convert].replace(0, np.nan)



    merged_updated_df.loc[(merged_updated_df['record_type'] == 'RANGE') & (merged_updated_df['action'] == 'ADD'), ['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']] = np.nan
    #full_match_output_df.loc[(merged_updated_df['record_type'] == 'RANGE') & (merged_updated_df['action'] == 'ADD'), ['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']] = ""
    different_ranges_alma_output_df.loc[(merged_updated_df['record_type'] == 'RANGE') & (merged_updated_df['action'] == 'ADD'), ['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']] = np.nan

    full_match_output_df.to_csv('Output/Full Match Final.csv', index=False)
    merged_updated_df.to_csv('Output/Update Final.csv', index=False)
    
    #different_ranges_docline_output_df.to_csv('Output/Different Ranges Docline Final.csv', index=False)
    different_ranges_alma_output_df.to_csv('Output/Different Ranges Alma Final.csv', index=False)
    #
    #full_match_output_df[['begin_year', 'end_year', 'begin_volume', 'end_volume', 'embargo_period', 'limited_retention_period']] = full_match_output_df[['begin_year', 'end_year', 'begin_volume', 'end_volume', 'embargo_period', 'limited_retention_period']].astype('Int64')
    #different_ranges_alma_output_df[['begin_year', 'end_year', 'begin_volume', 'end_volume', 'embargo_period', 'limited_retention_period']] = different_ranges_alma_output_df[['begin_year', 'end_year', 'begin_volume', 'end_volume', 'embargo_period', 'limited_retention_period']].astype('Int64')

    #full_match_output_df[['begin_year', 'end_year', 'begin_volume', 'end_volume', 'embargo_period', 'limited_retention_period']] = full_match_output_df[['begin_year', 'end_year', 'begin_volume', 'end_volume', 'embargo_period', 'limited_retention_period']].astype('Int64')
    #different_ranges_alma_output_df[['begin_year', 'end_year', 'begin_volume', 'end_volume', 'embargo_period', 'limited_retention_period']] = different_ranges_alma_output_df[['begin_year', 'end_year', 'begin_volume', 'end_volume', 'embargo_period', 'limited_retention_period']].astype('Int64')

    # merged_updated_df[['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']] = merged_updated_df[['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']].astype('str')
    # full_match_output_df[['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']] = full_match_output_df[['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']].astype('str')
    # different_ranges_alma_output_df[['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']] = merged_updated_df[['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']].astype('str')

    #merged_updated_df[['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']] = merged_updated_df[['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']].astype('str')
    #full_match_output_df[['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']] = full_match_output_df[['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']].astype('str')
    #different_ranges_alma_output_df[['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']] = different_ranges_alma_output_df[['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']].astype('str')



    #full_match_output_df.loc[(full_match_output_df['record_type'] == 'RANGE') & (full_match_output_df['action'] == 'ADD'), ['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']] = ""
    #different_ranges_alma_output_df.loc[(different_ranges_alma_output_df['record_type'] == 'RANGE') & (different_ranges_alma_output_df['action'] == 'ADD'), ['serial_title', 'nlm_unique_id', 'holdings_format', 'issns', 'currently_received', 'retention_policy', 'limited_retention_type', 'has_epub_ahead_of_print', 'has_supplements', 'ignore_warnings', 'last_modified']] = ""


    #merged_updated_df.to_csv('Output/Update Final1.csv', index=False)
    #full_match_output_df.to_csv('Output/Full Match Final1.csv', index=False)
    #different_ranges_alma_output_df.to_csv('Output/Different Ranges Alma1.csv', index=False)


    # ---------------------------------------------------------------------
    # Final output (single write pass)
    #   - Strip column-name whitespace (prevents "libid " sneaking through)
    #   - Drop libid from *all* deliverables (per request)
    #   - Ensure no helper index columns (level_0/index) leak into outputs
    #   - Write No-Dates as a real XLSX (not CSV with .xlsx extension)
    # ---------------------------------------------------------------------
    def _normalize_and_drop(df, drop_cols):
        if df is None or not hasattr(df, "columns"):
            return df
        # normalize column labels
        df = df.copy()
        df.columns = df.columns.astype(str).str.strip()
        # drop known junk + requested drops
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        return df

    DROP_COLS = ["libid", "index", "level_0"]

    deleted_output_df = _normalize_and_drop(deleted_output_df, DROP_COLS)
    full_match_output_df = _normalize_and_drop(full_match_output_df, DROP_COLS)
    merged_updated_df = _normalize_and_drop(merged_updated_df, DROP_COLS)
    different_ranges_docline_output_df = _normalize_and_drop(different_ranges_docline_output_df, DROP_COLS)
    different_ranges_alma_output_df = _normalize_and_drop(different_ranges_alma_output_df, DROP_COLS)
    in_docline_only_preserve_df = _normalize_and_drop(in_docline_only_preserve_df, DROP_COLS)
    add_df = _normalize_and_drop(add_df, DROP_COLS)
    no_dates_df = _normalize_and_drop(no_dates_df, DROP_COLS)

    error_missing_begin_year_df = _normalize_and_drop(error_missing_begin_year_df, DROP_COLS)

    add_df['ignore_warnings'] = "Yes"
    full_match_output_df['ignore_warnings'] = "Yes"
    different_ranges_alma_output_df['ignore_warnings'] = "Yes"
    different_ranges_docline_output_df['ignore_warnings'] = "Yes"
    merged_updated_df['ignore_warnings'] = "Yes"
        # Error output: missing begin_year in RANGE rows
    if error_missing_begin_year_df is not None and not error_missing_begin_year_df.empty:
        # Ensure Error Message column exists
        if "Error Message" not in error_missing_begin_year_df.columns:
            error_missing_begin_year_df["Error Message"] = (
                "Could not read begin date of portfolio from coverage statement"
            )

        # Match Add/Update column order, then append Error Message
        base_cols = list(add_df.columns)  # Add Final and Update Final share the same schema here
        ordered_cols = [c for c in base_cols if c in error_missing_begin_year_df.columns]
        error_out = error_missing_begin_year_df[ordered_cols + ["Error Message"]]

        error_out.to_csv("Output/Error Final.csv", index=False)

    # Core CSV outputs
    add_df.to_csv("Output/Add Final.csv", index=False)
    full_match_output_df.to_csv("Output/Full Match Final.csv", index=False)
    merged_updated_df.to_csv("Output/Update Final.csv", index=False)
    different_ranges_docline_output_df.to_csv("Output/Different Ranges Docline Final.csv", index=False)
    different_ranges_alma_output_df.to_csv("Output/Different Ranges Alma Final.csv", index=False)

    # Deleted output (this file existed earlier in the workflow; always overwrite here)
    deleted_output_df.to_csv(
        "Output/Delete Final - Either Withdrawn from Alma or ILL Not Allowed for E-Resources.csv",
        index=False,
    )

    # Preserve list + counts
    in_docline_only_preserve_df = in_docline_only_preserve_df.reset_index(drop=True)
    counts_df = pd.concat(
        [
            counts_df,
            pd.DataFrame(
                {
                    "Set": "In Docline Only Keep",
                    "Number of Rows": len(in_docline_only_preserve_df),
                    "Number of NLM Unique IDs": len(pd.unique(in_docline_only_preserve_df["nlm_unique_id"])),
                },
                index=[0],
            ),
        ],
        ignore_index=True,
    )
    in_docline_only_preserve_df.to_csv("Output/In Docline Only Preserve Final.csv", index=False)

    # No-dates output (real XLSX)
    try:
        with pd.ExcelWriter("Output/No Dates in Update Table.xlsx", engine="openpyxl") as xw:
            no_dates_df.to_excel(xw, sheet_name="No Dates", index=False)
    except Exception as e:
        # Fallback: write CSV next to it if Excel output fails for any reason
        no_dates_df.to_csv("Output/No Dates in Update Table.csv", index=False)
        print(f"WARNING: could not write No Dates XLSX ({e}); wrote CSV instead.")

    # Counts
    counts_df.to_excel("Output/Counts.xlsx", index=False)

