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


def parse_xml_chunked(file_path, tag):
    for event, elem in ET.iterparse(file_path, events=("start", "end")):
        if event == "end" and elem.tag == tag:
            yield elem
            elem.clear()
def getNLMData(nlm_choice):
    #43-366
    start_time = time.time()
    filename = ""
    if str(nlm_choice) == "1":
        webpage = requests.get("https://ftp.nlm.nih.gov/projects/serfilebase")

        soup = BeautifulSoup(webpage.content, "html.parser")
        dom = etree.HTML(str(soup))
        # this loops through the links in this directory
        # the last link will be the one that is assigned
        # to the variable a in the last iteration of the loop
        for a in dom.xpath('//ul/li/a/@href'):

            a = a

        latest_file = a


        full_nlm_serials_data_mrc = requests.get("https://ftp.nlm.nih.gov/projects/serfilebase/" + str(latest_file))
        filename = "marc_output_via_bs4_and_requests.mrc"
        full_filename = oDir + "/" + filename
        mrc_output = open(oDir + "/" + filename, "wb")
        mrc_output.write(full_nlm_serials_data_mrc.content)
    elif nlm_choice == "2":

        files = glob.glob('NLM/*.csv', recursive = True)

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
                    # print("got into oclc loop")
                    # print(oclc_035_tag['a'])
                    dict = oclc_035_tag.subfields_as_dict()
                    oclc_list = []
                    for key, value in dict.items():

                        for each_value in value:

                            if re.search("\(OCoLC\)", each_value):
                                oclc_list.append(re.sub(r'\(OCoLC\)\s*(\d+)', r'\1', each_value))
                            #     if y == 0:
                            #         oclc_number = ""
                            #         print("NLM Unique ID" + nlm_unique_id)
                            #         print("oclc number match")
                            #         oclc_number = re.sub(r'\(OCoLC\)\s*(\d+)', r'\1', each_value)
                            #         print("oclc number parsed" + oclc_number)
                            #     else:
                            #         print("NLM Unique ID" + nlm_unique_id)
                            #         print("oclc number match")
                            #         oclc_number += oclc_number + ";" + re.sub(r'\(OCoLC\)\s*(\d+)', r'\1', each_value)
                            #         print("oclc number parsed" + oclc_number)
                            # y += 1
                    #dedupe
                    oclc_list = list(dict.fromkeys(oclc_list))
                    oclc_number = ";".join(oclc_list)

                #     print("OCLC list" + oclc_number)
                # if x == 50:
                #     sys.exit()

                # except:
                #     oclc_number = "None"

                try:
                    for record_022 in record.get_fields('022'):
                        issn = record_022['a']
                        #print("got into issn for loop")
                        #print(record_022)
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

                    print(csv_data[x])
                x += 1



        csv_filename = "Processing/journal_data.csv"
        with open(csv_filename, encoding='utf-8', mode="w", newline="\n") as csv_file:
            writer = csv.writer(csv_file, delimiter='\t')
            writer.writerow(csv_headers)
            writer.writerows(csv_data)
        #log_file = open("Output/Execution time of docline script.txt", "w+")
        #first_time = time.time()
        #print("--- %s minutes to complete MRC NLM transformation ---\n" % ((first_time - start_time)/60))
        #
        #log_file.write("--- %s minutes to complete MRC NLM transformation ---\n" % (first_time - start_time))
        # #
        #
        #
        # print(f"CSV data has been written to {csv_filename}")
        #
        #
        #

def getAnalyticsData():
    # # files = glob.glob('Analytics/*', recursive = True)
    # #
    # # analytics_filename = files[0]
    # #
    # # sys.exit()
    #
    #
    #
    #
    files = glob.glob('Analytics/*.csv', recursive = True)
    #
    analytics_filename = files[0]
    #
    # second_time = time.time()
    # print("--- %s minutes to complete Alma Analytics retrieval and parsing ---\n" % ((second_time - first_time)/60))
    #
    # log_file.write("--- %s minutes to complete Alma Analytics retrieval and parsing ---\n" % (second_time - first_time))
    #
    #
    # # sys.exit()
    # #
    # # files = glob.glob('Analytics/*', recursive = True)
    # #
    # # analytics_filename = files[0]
    #
    #
    #

    return analytics_filename

def groupAndMergeAlmaAndDocline(analytics_filename):
    df = pd.read_csv(analytics_filename, dtype={"MMS Id": "str", 'Network Number (OCoLC)': 'str'}, encoding='utf-8')



    print("read")

    # 1. group by

    grouped_df = df.groupby(["Title", "MMS Id", "Network Number (OCoLC)", "ISSN", "Lifecycle", "Electronic or Physical"])["Coverage Information Combined"].apply(lambda x: ';'.join(x)).reset_index()
    print("complete groupby")
    # 1. Explode on OCLC
    #grouped_df['Network Number (OCoLC)'] = grouped_df['Network Number (OCoLC)'].astype(str)
    exploded_oclc_df = grouped_df.copy()
    exploded_oclc_df['OCLC'] = exploded_oclc_df['Network Number (OCoLC)'].str.split(';')

    exploded_oclc_df = exploded_oclc_df.explode('OCLC').reset_index(drop=True)
    # print(exploded_oclc_df)


    exploded_oclc_df['OCLC'] = exploded_oclc_df['OCLC'].str.strip()
    pd.set_option('display.max_columns', None)
    # print(exploded_oclc_df.head(100))

    exploded_oclc_df['OCLC'] = exploded_oclc_df['OCLC'].astype('string')

    exploded_oclc_df.to_csv("Processing/Exploded.csv", index=False)
    analytics_ocolc_df = exploded_oclc_df[exploded_oclc_df['OCLC'].str.contains("OCoLC")]

    analytics_ocolc_df.to_csv("Processing/A-Has OCLC.csv", index=False)
    analytics_no_ocolc_df = exploded_oclc_df[~exploded_oclc_df['OCLC'].notna()]

    analytics_no_ocolc_df.to_excel("Processing/Analytics No OCoLC.xlsx", index=False)





    #analytics_ocolc_df['OCLC'] = analytics_ocolc_df['OCLC'].apply(lambda x: re.sub(r'\(OCoLC\)[a-z\s]*(\d+)', r'\1', x))
    # print(analytics_ocolc_df)
    analytics_ocolc_df.to_csv("Processing/A-Has OCLC.csv", index=False)
    # 2. Group and concatenate
    print("complete explode")
    #grouped_df = analytics_ocolc_df.groupby(["Title", "MMS Id", "OCLC", "ISSN", "Bibliographic Lifecycle"])["Coverage Information Combined"].apply(lambda x: ';'.join(x)).reset_index()







    # 3. explode Analytics on ISSN
    # this is a separate set than the OCLC set above, so the
    # merge will be done separately and then concatenated

    exploded_issn_df = grouped_df.copy()
    exploded_issn_df['ISSN'] = grouped_df['ISSN'].str.split('; ')
    exploded_issn_df = exploded_issn_df.explode('ISSN').reset_index(drop=True)


    exploded_issn_df['ISSN'] = exploded_issn_df['ISSN'].str.strip()


    exploded_issn_df = exploded_issn_df[exploded_issn_df['ISSN'].notna() & exploded_issn_df['ISSN'] != 'None']

    # print(exploded_issn_df)

    # 4. Merge with NLM data, mostly to get NLM Unique ID for the Alma data
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

    # 5. restrict NLM subset to those that have ISSN
    # as noted above for the Alma data, this is a separate set
    # that will be matched separately and then
    # concatenated together with the OCLC matches

    issn_journal_data_df = journal_data_df.copy()





    issn_journal_data_df = issn_journal_data_df.loc[((issn_journal_data_df['ISSN'] != "None") | (issn_journal_data_df['Electronic_ISSN'] != "None") | (issn_journal_data_df['Print_ISSN'] != "None"))]





    # 6. merge
    # this merges on matches, and then concatenates all the merges
    # together into a single dataframe that can go
    # into the following Docline parsing and
    # comparison processes


    merged_oclc_df = analytics_ocolc_df.merge(exploded_oclc_journal_data_df, left_on='OCLC', right_on="OCLC_Number", how='inner')

    merged_oclc_df = merged_oclc_df.rename(columns={"ISSN_y": 'ISSN'})
    merged_issn_df = exploded_issn_df.merge(issn_journal_data_df, on='ISSN', how='inner')

    merged_electronic_issn_df = exploded_issn_df.merge(issn_journal_data_df, how='inner', left_on='ISSN', right_on='Electronic_ISSN')
    merged_print_issn_df = exploded_issn_df.merge(issn_journal_data_df, how='inner', left_on='ISSN', right_on='Print_ISSN')
    # put print in here too

    merged_oclc_df.to_excel('Processing/Merged OCLC.xlsx', index=False)
    merged_issn_df.to_excel('Processing/Merged ISSN.xlsx', index=False)
    merged_electronic_issn_df.to_excel('Processing/Merged Electronic ISSN.xlsx', index=False)
    merged_print_issn_df.to_excel('Processing/Merged Print ISSN.xlsx', index=False)
    merged_df = merged_oclc_df.copy()

    merged_df = pd.concat([merged_oclc_df, merged_issn_df, merged_electronic_issn_df]).reset_index()

    merged_df = merged_df.drop('ISSN_y', axis=1)

    merged_df.to_excel("Processing/Merged All.xlsx", index=False)
    print(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['NLM_Unique_ID', 'Electronic or Physical'])

    merged_df.to_excel("Processing/Merged Deduped All.xlsx", index=False)

    # 7. Save the merged data
    merged_df.to_excel("Processing/merged_journals_data2.xlsx", index=False)

    merged_df.to_csv("Processing/merged_journal_data2.csv", index=False)

    print("Number of records in match table")
    print(len(merged_df))


    return merged_df

def convert(merged_df, docline_df):


    alma_nlm_merge_df = pd.DataFrame(columns=docline_df.columns)

    alma_nlm_merge_df['Bibliographic Lifecycle'] = ""


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
        main_row = {
            'Bibliographic Lifecycle': row['Lifecycle'],
            'action': '',
            'record_type': 'HOLDING',
            'libid': secrets_local.libid,
            'serial_title': row['Title_x'],
            'nlm_unique_id': row['NLM_Unique_ID'],
            'holdings_format': row['Electronic or Physical'],
            'begin_volume': '',
            'end_volume': '',
            'begin_year': '',
            'end_year': '',
            'issns': row['ISSN_x'].replace(';', ','),
            'currently_received': 'No', #if 'until' in str(row['Coverage Information Combined']) else 'Yes',
            'retention_policy': secrets_local.retention_policy,
            'limited_retention_period': secrets_local.limited_retention_period,
            'embargo_period': secrets_local.embargo_period,
            'limited_retention_type': secrets_local.limited_retention_type,
            'has_epub_ahead_of_print': secrets_local.has_epub_ahead_of_print,
            'has_supplements': secrets_local.has_supplements,
            'ignore_warnings': secrets_local.ignore_warnings,
            'last_modified': ''#pd.to_datetime('today').strftime('%Y-%b-%d')
        }
        alma_nlm_merge_df = pd.concat([alma_nlm_merge_df, pd.DataFrame(main_row, index=[0])])

        # Create additional HOLDING rows for coverage data
        coverage_data = str(row['Coverage Information Combined']).split(';')

        if x % 100 == 0:
            print(str(row['Title_x']) + "--" + str(row['ISSN_x']))

        x += 1

        coverage_data_output = []
        for coverage in coverage_data:
            begin_year = None
            end_year = None
            begin_volume = None
            end_volume = None
            if 'from' in coverage and 'until' in coverage:
                beginning = coverage.split('from')[-1].split('until')[0].strip()#.split(r'[-\s \\]')[-1].strip()

                begin_year = re.sub(r'^[^:]*?(\d{4}).*$', r'\1', beginning)

                if 'volume' in beginning:
                    begin_volume = re.sub(r'^.*?volume\:\s+(\d+).*$', r'\1', beginning)


                end = coverage.split('until')[-1]

                end_year = re.sub(r'^[^:]*?(\d{4}).*$', r'\1', end)

                if 'volume' in end:
                    end_volume = re.sub(r'^.*?volume\:\s+(\d+).*$', r'\1', end)

            elif 'from' in coverage:

                beginning = coverage.split('from')[-1].strip()
                begin_year = re.sub(r'^[^:]*?(\d{4}).*$', r'\1', beginning)
                if 'volume' in beginning:
                    begin_volume = re.sub(r'^.*?volume\:\s+(\d+).*$', r'\1', beginning)
                else:
                    begin_volume = None



            coverage_row = {
                'Bibliographic Lifecycle': row['Lifecycle'],
                'action': '',
                'record_type': 'RANGE',
                'libid': secrets_local.libid,
                'serial_title': row['Title_x'],
                'nlm_unique_id': row['NLM_Unique_ID'],
                'holdings_format': row['Electronic or Physical'],
                'begin_volume': begin_volume,
                'end_volume': end_volume,
                'begin_year': begin_year,
                'end_year': end_year,
                'issns': row['ISSN_x'].replace(';', ','),
                'currently_received': 'No', #if 'until' in str(row['Coverage Information Combined']) else 'Yes',
                'retention_policy': secrets_local.retention_policy,
                'limited_retention_period': secrets_local.limited_retention_period,
                'embargo_period': secrets_local.embargo_period,
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
    columns_to_update = ['nlm_unique_id', 'serial_title', 'libid', 'holdings_format', 'issns', 'currently_received', 'retention_policy',
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
    alma_nlm_merge_df['nlm_unique_id'] = alma_nlm_merge_df['nlm_unique_id'].astype(str)



    alma_nlm_merge_df.to_excel("Processing/Alma Docline output.xlsx", index=False)
    alma_nlm_merge_df.to_csv("Processing/Alma Docline output.csv", quoting=csv.QUOTE_ALL, index=False)
    #

    alma_nlm_merge_df = pd.read_csv("Processing/Alma Docline output.csv", engine='python', dtype={"nlm_unique_id": str})


    alma_nlm_merge_df = alma_nlm_merge_df[(~alma_nlm_merge_df.isnull()['nlm_unique_id'] & alma_nlm_merge_df['record_type'].isin(['HOLDING'])) | ((alma_nlm_merge_df['record_type'].isin(['RANGE'])) & (~alma_nlm_merge_df.isnull()['begin_year'] | ~alma_nlm_merge_df.isnull()['begin_volume']))].reset_index(drop=True)





    alma_nlm_merge_df[['begin_year', 'end_year', 'begin_volume', 'end_volume']] = alma_nlm_merge_df[['begin_year', 'end_year', 'begin_volume', 'end_volume']].astype('float')
    alma_nlm_merge_df[['begin_year', 'end_year', 'begin_volume', 'end_volume']] = alma_nlm_merge_df[['begin_year', 'end_year', 'begin_volume', 'end_volume']].astype('Int64')

    # note so that the docline data can be treated as a real, valid table
    # data from the HOLDING line is copied over on to the RANGE line,
    # a relationship that in Docline format is only represented by the RANGE data
    # immediately following the HOLDING line.
    # this way they can be sorted

    files_docline = glob.glob('Docline/*', recursive = True)

    docline_filename = files_docline[0]
    # Load the Docline CSV file
    docline_df = pd.read_csv(docline_filename, dtype={'begin_year': 'Int64', 'end_year': 'Int64', 'begin_volume': 'Int64', 'end_volume': 'Int64', 'nlm_unique_id': 'str'}, engine='python')


    #docline_df.to_csv('Processing/Test Docline DF.csv', index=False)
    # Function to apply values in RANGE rows associated with holdings
    # simply by order, with al data from the holding row
    #
    # this was largely created by ChatGPT 4 with some tweaking

    # Apply the function to the DataFrame
    existing_docline_df = propagate_nlm_unique_id_and_libid_values(docline_df.copy())

    # print(existing_docline_df)

    # sys.exit()
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

    list = string.split("; ")

    if "<NA>" in list:
        list.remove("<NA>")

    string = "; ".join(list)

    return(string)
def merge_intervals_optimized(df):
    """
    An optimized function to merge overlapping intervals for the entire dataframe.
    """
    # Sort the dataframe by nlm_unique_id and begin_year
    #df.sort_values(by=['nlm_unique_id', 'begin_year'], inplace=True)

    # Replace NaN in end_year with a large number to represent 'indefinite'
    df['end_year'].fillna(float('inf'), inplace=True)

    merged_intervals = []
    current_start = current_end = None
    current_id = None
    currently_received = "No"
    for index, row in df.iterrows():
        if row['end_year'] > 3000:
            currently_received = "Yes"
            df['currently_received'] = currently_received
        else:
            df['currently_received'] = "No"
        if row['nlm_unique_id'] != current_id:
            # Save the previous interval if it exists
            if current_id is not None:
                merged_intervals.append([current_id, current_start, None if current_end == float('inf') else current_end])
            # Start a new interval
            current_id, current_start, current_end = row['nlm_unique_id'], row['begin_year'], row['end_year']
        else:
            # If the current interval overlaps with the ongoing one, merge them
            if row['begin_year'] <= current_end:
                current_end = max(current_end, row['end_year'])
            else:
                # Save the previous interval and start a new one
                merged_intervals.append([current_id, current_start, None if current_end == float('inf') else current_end])
                current_start, current_end = row['begin_year'], row['end_year']

    # Add the last interval
    merged_intervals.append([current_id, current_start, None if current_end == float('inf') else current_end])

    # Create a DataFrame from the merged intervals
    merged_df = pd.DataFrame(merged_intervals, columns=['nlm_unique_id', 'begin_year', 'end_year'])
    return merged_df
def merge(alma_nlm_merge_df, existing_docline_df):
    #####################################################
    #####################################################
    ####    This is the final merge: existing Docline
    ####    against the dataframe of parsed Alma holdings compared
    ####    against NLM data
    ####
    ####    Method:
    ####        - generally by nlm_unique_id, substracting
    ####          the "NLM_" prefix from the Docline data for the purpose of the match
    ####        - before the merge, we need to separate
    ####          "Deleted" from In "Repository" into separate dataframes
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
    deleted_alma_df = alma_nlm_merge_df.copy()
    #
    deleted_alma_df = deleted_alma_df[deleted_alma_df['Bibliographic Lifecycle'] == 'Deleted']
    #
    current_alma_df = alma_nlm_merge_df.copy()

    current_alma_df = current_alma_df[~current_alma_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(deleted_alma_df.set_index(['nlm_unique_id', 'holdings_format']).index)]

    deleted_alma_df = deleted_alma_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'begin_volume', 'end_year', 'end_volume'], ascending = [True, True, True, True, True, True], na_position = 'first')

    current_alma_df = current_alma_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'begin_volume', 'end_year', 'end_volume'], ascending = [True, True, True, True, True, True], na_position = 'first')

    existing_docline_df_for_compare = existing_docline_df.copy()
    existing_docline_df_for_compare['nlm_unique_id'] = existing_docline_df_for_compare['nlm_unique_id'].apply(lambda x: x.replace("NLM_", ""))
    # print(existing_docline_df_for_compare)
    deleted_output_df = deleted_alma_df.copy()

    deleted_output_df = deleted_output_df[deleted_output_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(existing_docline_df_for_compare.set_index(['nlm_unique_id', 'holdings_format']).index)]

    deleted_output_df.to_excel("Processing/Deleted Output DF.xlsx", index=False)
    existing_docline_df_for_compare.to_excel('Processing/Existing Docline for Compare.xlsx', index=False)
    # print(deleted_output_df)
    #
    # sys.exit()
    # cases where the NLM unique ID and holdings format combination
    # between the Alma data
    # (with the NLM Unique ID coming from the NLM catalog as noted above)
    #  are more complicated
    # first, there are those records that match exactly what is in Docline,
    # including RANGE.  These can be discluded from the import as they remain the same

    # things that match on NLM Unique ID and holdings format,
    # but differ in either range or other holdings field need to be updated
    # This requires that the script first submit a DELETE command for what
    # you have in Docline now,
    # and then ADD for the new holding and Ranges,
    # basically a drawn out UPDATE, which doesn't exist in Docline processing

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
    # and then later in the file add the Alma data as an ADD



    current_alma_df = current_alma_df.reset_index()
    existing_docline_df_for_compare = existing_docline_df_for_compare.reset_index()
    current_alma_df = current_alma_df.drop('Bibliographic Lifecycle', axis=1)


    current_alma_df = current_alma_df.drop('index', axis=1)
    existing_docline_df_for_compare = existing_docline_df_for_compare.drop('index', axis=1)
    matched_base_df_alma_side = current_alma_df[current_alma_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(existing_docline_df_for_compare.set_index(['nlm_unique_id', 'holdings_format']).index)]
    matched_base_df_docline_side = existing_docline_df_for_compare[existing_docline_df_for_compare.set_index(['nlm_unique_id', 'holdings_format']).index.isin(current_alma_df.set_index(['nlm_unique_id', 'holdings_format']).index)]
    pd.set_option('display.max_columns', None)
    #print(matched_base_df_alma_side)
    #print(matched_base_df_docline_side)

    matched_base_df_alma_side.to_csv('Analysis/matched_base_df_alma_side.csv', index=False)
    matched_base_df_docline_side.to_csv('Analysis/matched_base_df_docline_side.csv', index=False)




    # These are the Alma records to add to Docline, because they are not
    # in current Docline holdings by NLM Unique ID
    add_df = matched_base_df_alma_side.copy()
    add_df = current_alma_df[~current_alma_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(existing_docline_df_for_compare.set_index(['nlm_unique_id', 'holdings_format']).index)]

    # These are the Docline records that did not match to NLM-enriched Alma
    # data by NLM unique ID.  They should be reviewed manually
    in_docline_only_preserve_df = existing_docline_df_for_compare.copy()
    in_docline_only_preserve_df = existing_docline_df_for_compare[~existing_docline_df_for_compare.set_index(['nlm_unique_id', 'holdings_format']).index.isin(alma_nlm_merge_df.set_index(['nlm_unique_id', 'holdings_format']).index)]


    all_columns = existing_docline_df_for_compare.columns


    aggregate_columns = []
    group_by_columns = []
    for column in all_columns:
        if column in ['begin_year', 'end_year']:
            aggregate_columns.append(column)
        elif column == 'Bibliographic Lifecycle' or column == 'last_modified' or column == 'issns' or column == 'begin_volume' or column == 'end_volume':
            continue
        else:
            group_by_columns.append(column)


    all_columns = all_columns.tolist()

    all_columns.remove('last_modified')
    all_columns.remove('issns')
    all_columns.remove('Bibliographic Lifecycle')
    all_columns.remove('begin_volume')
    all_columns.remove('end_volume')




    # in this section, once the range columns and rows of
    # both the Alma-NLM data set and the existing Docline set
    # have been sorted in 6-tiered sort noted above,
    # all the range rows are rolled up into the holding
    # rows such that each title should appear only once
    #
    # With this in place, the whole rows (minus modified date and a few other fields)
    # from the Docline-formatted Alma set can be compared with whole rows from
    #the current Docline set to determine if the holdings *and* ranges are a complete match,
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
    #



    # dealing with date and volume columns that are float because they
    # contain numbers and NaN
    # https://stackoverflow.com/questions/60024262/error-converting-object-string-to-int32-typeerror-object-cannot-be-converted


    current_alma_df[aggregate_columns] = current_alma_df[aggregate_columns].astype('float')
    existing_docline_df_for_compare[aggregate_columns] = existing_docline_df_for_compare[aggregate_columns].astype('float')

    current_alma_df[aggregate_columns] = current_alma_df[aggregate_columns].astype('Int64')
    existing_docline_df_for_compare[aggregate_columns] = existing_docline_df_for_compare[aggregate_columns].astype('Int64')



    # note the group by columns being those fields in the holding row,
    # and the aggregate columns being those things in range, that are being rolled up
    existing_docline_for_compare_agg_df = existing_docline_df_for_compare.copy()
    matched_nlm_alma_df_for_compare_agg = current_alma_df.copy()



    existing_docline_for_compare_agg_df = existing_docline_for_compare_agg_df.groupby(group_by_columns, dropna=False, as_index=False).agg({aggregate_columns[0]: lambda x: '; '.join(set(x.astype(str))), aggregate_columns[1]: lambda x: '; '.join(set(x.astype(str))), aggregate_columns[2]: lambda x: '; '.join(set(x.astype(str))), aggregate_columns[3]: lambda x: '; '.join(set(x.astype(str)))})
    matched_nlm_alma_df_for_compare_agg = matched_nlm_alma_df_for_compare_agg.groupby(group_by_columns, dropna=False, as_index=False).agg({aggregate_columns[0]: lambda x: '; '.join(set(x.astype(str))), aggregate_columns[1]: lambda x: '; '.join(set(x.astype(str))), aggregate_columns[2]: lambda x: '; '.join(set(x.astype(str))), aggregate_columns[3]: lambda x: '; '.join(set(x.astype(str)))})



    # in the rolled up range data, remove the appearance of "<NA>" in the stringified list
    matched_nlm_alma_df_for_compare_agg[['begin_year', 'end_year', 'begin_volume', 'end_volume']] = matched_nlm_alma_df_for_compare_agg[['begin_year', 'end_year', 'begin_volume', 'end_volume']].applymap(lambda x: removeNA(x))



    existing_docline_for_compare_agg_df.to_excel("Processing/existing_docline_for_compare_agg_df.xlsx", index=False)
    matched_nlm_alma_df_for_compare_agg.to_excel('Processing/matched_nlm_alma_df_for_compare_agg.xlsx', index=False)



    full_merge_on_just_nlm_id_and_format = pd.merge(existing_docline_for_compare_agg_df, matched_nlm_alma_df_for_compare_agg, how='inner', on=['nlm_unique_id', 'holdings_format'])

    full_merge_on_just_nlm_id_and_format.to_excel('Analysis/merged_matched_on_nlm_id_and_holding_format.xlsx', index=False)





    full_matched_for_compare_df = matched_nlm_alma_df_for_compare_agg.copy()
    full_matched_for_compare_df = full_matched_for_compare_df.reset_index()




    existing_docline_for_compare_agg_df = existing_docline_for_compare_agg_df.reset_index()

    # we don't need to include the nlm unique id or format criterion here
    # explicityl because all_columns includes this

    full_matched_for_compare_df = full_matched_for_compare_df[full_matched_for_compare_df.set_index(all_columns).index.isin(existing_docline_for_compare_agg_df.set_index(all_columns).index)]
    #
    #
    full_matched_for_compare_df.to_excel("Processing/full_matched_for_compare_df.xlsx", index=False)





    # These are the Alma records in which the NLM Unique ID matches,
    # but the entire aggregated row (from above) doesn't match,
    # i.e either the holding data or the range data is different.
    # if the holding data is different, it's possible that you've made some
    # different decisions about fields such as limited_retention_policy than you have in the past

    # as mentioned above, this is aggregated so it won't *be* the output
    # data but will be used to filter it
    different_ranges_alma_compare_df_agg = matched_nlm_alma_df_for_compare_agg.copy()


    different_ranges_alma_compare_df_agg = different_ranges_alma_compare_df_agg[(different_ranges_alma_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index.isin(existing_docline_for_compare_agg_df.set_index(['nlm_unique_id', 'holdings_format']).index)) & (~different_ranges_alma_compare_df_agg.set_index(all_columns).index.isin(existing_docline_for_compare_agg_df.set_index(all_columns).index))]

    # ditto to above, but among the current Docline set compared to Alma (other direction)
    different_ranges_docline_compare_df_agg = existing_docline_for_compare_agg_df.copy()
    #                                     different_ranges_alma_compare_df_agg[(different_ranges_alma_compare_df_agg.nlm_unique_id.isin(existing_docline_for_compare_agg_df.nlm_unique_id)) & (~different_ranges_alma_compare_df_agg.set_index(all_columns).index.isin(existing_docline_for_compare_agg_df.set_index(all_columns).index))]
    #different_ranges_docline_compare_df_agg = existing_docline_for_compare_agg_df[(existing_docline_for_compare_agg_df.nlm_unique_id.isin(matched_nlm_alma_df_for_compare_agg.nlm_unique_id)) & (~existing_docline_for_compare_agg_df.set_index(all_columns).index.isin(matched_nlm_alma_df_for_compare_agg.set_index(all_columns).index)) & (existing_docline_for_compare_agg_df['holdings_format'] == matched_nlm_alma_df_for_compare_agg['holdings_format'])]
    different_ranges_docline_compare_df_agg = different_ranges_docline_compare_df_agg[(different_ranges_docline_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index.isin(matched_nlm_alma_df_for_compare_agg.set_index(['nlm_unique_id', 'holdings_format']).index)) & (~different_ranges_docline_compare_df_agg.set_index(all_columns).index.isin(matched_nlm_alma_df_for_compare_agg.set_index(all_columns).index))]


    different_ranges_docline_compare_df_agg = different_ranges_docline_compare_df_agg.drop('index', axis=1)

    # print(existing_docline_df_for_compare)
    # print(different_ranges_docline_compare_df_agg)


    # now take the output produced with the iteration loops, and test it for
    # membership in the above dataframes processed for testing,
    # to get the final output dataframes
    full_match_output_df = alma_nlm_merge_df.copy()
    full_match_output_df = full_match_output_df.reset_index()
    full_match_output_df = full_match_output_df[full_match_output_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(full_matched_for_compare_df.set_index(['nlm_unique_id', 'holdings_format']).index)]

    counts_df = pd.DataFrame(columns=['Set', 'Number of Rows', 'Number of NLM Unique IDs'])


    # print("full match output")
    #
    # print(full_match_output_df)
    # print(type(full_match_output_df))
    # print("number of rows")
    # print(len(full_match_output_df))
    # print("unique NLM IDs")

    counts_df = pd.concat([counts_df, pd.DataFrame({'Set': 'Full Match', 'Number of Rows':len(full_match_output_df), 'Number of NLM Unique IDs': len(pd.unique(full_match_output_df['nlm_unique_id']))}, index=[0])])

    # print(len(pd.unique(full_match_output_df['nlm_unique_id'])))
    #
    # print(len(full_match_output_df['nlm_unique_id'].value_counts()))




    different_ranges_alma_output_df = alma_nlm_merge_df.copy()
    different_ranges_alma_output_df = different_ranges_alma_output_df[different_ranges_alma_output_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(different_ranges_alma_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index)]
    different_ranges_alma_output_df = different_ranges_alma_output_df.reset_index()
    counts_df = pd.concat([counts_df, pd.DataFrame({'Set': 'Alma Different Ranges', 'Number of Rows': len(different_ranges_alma_output_df), 'Number of NLM Unique IDs': len(pd.unique(different_ranges_alma_output_df['nlm_unique_id']))}, index=[0])])
    different_ranges_alma_output_df = different_ranges_alma_output_df.reset_index()




    different_ranges_docline_output_df = existing_docline_df_for_compare.copy()
    different_ranges_docline_output_df = existing_docline_df_for_compare[existing_docline_df_for_compare.set_index(['nlm_unique_id', 'holdings_format']).index.isin(different_ranges_docline_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index)]


    # print(different_ranges_docline_output_df)
    counts_df = pd.concat([counts_df, pd.DataFrame({'Set': 'Docline Different Ranges', 'Number of Rows': len(different_ranges_docline_output_df), 'Number of NLM Unique IDs': len(pd.unique(different_ranges_docline_output_df['nlm_unique_id']))}, index=[0])])




    add_df['nlm_unique_id'] = add_df['nlm_unique_id'].apply(lambda x: "NLM_" + x)
    deleted_output_df['nlm_unique_id'] = deleted_output_df['nlm_unique_id'].apply(lambda x: "NLM_" + x)
    deleted_output_df.to_excel('Processing/Deleted DF After Applying _NLM_ to NLM Unique ID.xlsx', index=False)
    full_match_output_df['nlm_unique_id'] = full_match_output_df['nlm_unique_id'].apply(lambda x: "NLM_" + x)
    different_ranges_alma_output_df['nlm_unique_id'] = different_ranges_alma_output_df['nlm_unique_id'].apply(lambda x: "NLM_" + x)
    different_ranges_docline_output_df['nlm_unique_id'] = different_ranges_docline_output_df['nlm_unique_id'].apply(lambda x: "NLM_" + x)

    add_df['action'] = 'ADD'
    deleted_output_df['action'] = 'DELETE'

    # do I need this?
    full_match_output_df['action'] = 'ADD'
    different_ranges_alma_output_df['action'] = 'ADD'
    different_ranges_docline_output_df['action']= 'DELETE'

    different_ranges_alma_output_df = different_ranges_alma_output_df.sort_values(by = ['nlm_unique_id', 'action', 'begin_year', 'end_year'], ascending = [True, False, True, True, True], na_position = 'first')
    different_ranges_docline_output_df = different_ranges_docline_output_df.sort_values(by = ['nlm_unique_id', 'action', 'begin_year', 'end_year'], ascending = [True, False, True, True, True], na_position = 'first')

    merged_updated_df = pd.concat([different_ranges_alma_output_df, different_ranges_docline_output_df])

    merged_updated_df = merged_updated_df.sort_values(by = ['nlm_unique_id', 'action', 'begin_year', 'end_year'], ascending = [True, False, True, True], na_position = 'first')
    #do I need this
    #different_ranges_alma_output_df.to_excel('Processing/Different Ranges Alma - After -Add- Action Entered.xlsx', index=False)
    #in_docline_only_preserve_df['action'] = 'ADD'

    merged_updated_df = merge_intervals_optimized(merged_updated_df)
    add_df = add_df.reset_index()
    counts_df = pd.concat([counts_df, pd.DataFrame({'Set': 'Alma Adds', 'Number of Rows': len(add_df), 'Number of NLM Unique IDs': len(pd.unique(add_df['nlm_unique_id']))}, index=[0])])


    # print("add_df length")
    # print(len(add_df))
    # print(len(pd.unique(add_df['nlm_unique_id'])))

    add_df = add_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'end_year'], ascending = [True, True, True, True], na_position = 'first')
    deleted_output_df = deleted_output_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'begin_volume', 'end_year'], ascending = [True, True, True, True], na_position = 'first')
    #deleted_output_df.to_excel('Processing/Deleted DF After Sort.xlsx', index=False)
    full_match_output_df = full_match_output_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'end_year'], ascending = [True, True, True, True])
    #different_ranges_alma_output_df = different_ranges_alma_output_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'end_year'], ascending = [True, True, True, True], na_position = 'first')
    #different_ranges_alma_output_df.to_excel('Processing/Add Alma After Sort.xlsx', index=False)
    #different_ranges_docline_output_df = different_ranges_docline_output_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'end_year'], ascending = [True, True, True, True], na_position = 'first')
    in_docline_only_preserve_df = in_docline_only_preserve_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'end_year'], ascending = [True, True, True, True], na_position = 'first')
    # print("\n\nbefore reset index\n\n")
    # print(different_ranges_alma_output_df)
    different_ranges_alma_output_df = different_ranges_alma_output_df.set_index('index')#reset_index()
    # print("\n\nafter reset index\n\n")
    # print(different_ranges_alma_output_df)


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
    # try:
    #     different_ranges_alma_output_df = different_ranges_alma_output_df.drop('index', axis=1)
    #
    # except:
    #     print('index already removed')
    # print("\n\nbefore remove level 0\n\n")
    #
    # print(different_ranges_alma_output_df)

    try:
        merged_updated_df = merged_updated_df.drop('level_0', axis=1)
    except:
        print("level 0 already removed")
    # print("\n\nafter remove level 0 and before remove bibliographic lifecycle\n\n")
    #
    # print(different_ranges_alma_output_df)
    try:
        merged_updated_df = merged_updated_df.drop('Bibliographic Lifecycle', axis=1)

    except:
        "Bibliographic Lifecycle column already removed"
    # print("\n\nafter remove bibliographic lifecycle\n\n")
    #
    # print(different_ranges_alma_output_df)

    # try:
    #     different_ranges_docline_output_df = different_ranges_docline_output_df.drop('index', axis=1)
    #
    # except:
    #     print('index already removed')
    #
    # try:
    #     in_docline_only_preserve_df = in_docline_only_preserve_df.drop('index', axis=1)
    #
    # except:
        # print('index already removed')
    add_df.to_csv('Output/Add Final.csv', index=False)


    # different_ranges_alma_output_df = different_ranges_alma_output_df.reset_index()
    try:
        merged_updated_df = merged_updated_df.drop('index', axis=1)

    except:
        "index column already removed"
    # deleted_output_df = deleted_output_df.reset_index()
    counts_df = pd.concat([counts_df, pd.DataFrame({'Set': 'Deleted from Alma', 'Number of Rows': len(deleted_output_df), 'Number of NLM Unique IDs': len(pd.unique(deleted_output_df['nlm_unique_id']))}, index=[0])])

    deleted_output_df.to_csv('Output/Delete Final.csv', index=False)
    full_match_output_df.to_csv('Output/Full Match Final.csv', index=False)
    merged_updated_df.to_csv('Output/Update Final.csv', index=False)
    #different_ranges_docline_output_df.to_csv('Output/Different Ranges Docline Final.csv', index=False)

    in_docline_only_preserve_df = in_docline_only_preserve_df.reset_index()
    counts_df = pd.concat([counts_df, pd.DataFrame({'Set': 'In Docline Only Keep', 'Number of Rows': len(in_docline_only_preserve_df), 'Number of NLM Unique IDs': len(pd.unique(in_docline_only_preserve_df['nlm_unique_id']))}, index=[0])])

    try:
        in_docline_only_preserve_df = in_docline_only_preserve_df.drop('index', axis=1)

    except:
        print('index already removed')
    in_docline_only_preserve_df.to_csv('Output/In Docline Only Preserve Final.csv', index=False)

    counts_df.to_excel('Output/Counts.xlsx', index=False)
