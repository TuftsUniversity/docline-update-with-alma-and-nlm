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
from datetime import datetime

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
def removeNA(string):

    list = string.split("; ")

    if "<NA>" in list:
        list.remove("<NA>")

    string = "; ".join(list)

    return(string)

def apply_currently_received(updated_df):
    updated_HOLDING_df = updated_df.copy()
    updated_HOLDING_df = updated_df[updated_df['record_type'] == "HOLDING"]
    updated_RANGE_df = updated_df.copy()
    updated_RANGE_df = updated_df[updated_df['record_type'] == "RANGE"]


    for index, row in updated_HOLDING_df.iterrows():
        nlm_left = row['nlm_unique_id']
        holdings_format_left = row['holdings_format']
        action_left = row['action']
        rows_df = updated_RANGE_df[(updated_RANGE_df['nlm_unique_id'] == nlm_left) & (updated_RANGE_df['holdings_format'] == holdings_format_left) & (updated_RANGE_df['action'] == action_left)]
        currently_received_list = rows_df['currently_received'].tolist()
        assignment_value = "No"
        for c_r in currently_received_list:
            if c_r == "Yes":
                assignment_value = "Yes"
                break
        if assignment_value != "Yes":
            assignment_value = "No"
        updated_df.loc[(updated_df['nlm_unique_id'] == nlm_left) & (updated_df['holdings_format'] == holdings_format_left) & (updated_df['action'] == action_left) & (updated_df['record_type'] == "HOLDING"), 'currently_received'] = assignment_value

    return updated_df

def merge_intervals_optimized(df):
    # Sort the DataFrame
    df.sort_values(by=['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year'], ascending=[True, True, False, True, True, True], inplace=True)
    # Using 10000 to represent 'indefinite'
    df['end_year'].fillna(10000, inplace=True)

    output_df = pd.DataFrame(columns=df.columns)
    current_row = None

    df = df.fillna(np.nan)
    for _, row in df.iterrows():
        if row['record_type'] == 'HOLDING':
            output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)
        elif row['record_type'] == 'RANGE':


            if current_row is not None:
                # if row['embargo_period']) == current_row['embargo_period'] and
                if row['nlm_unique_id'] != current_row['nlm_unique_id'] or \
                    row['holdings_format'] != current_row['holdings_format']:
                    output_df = pd.concat([output_df, pd.DataFrame([current_row])], ignore_index=True)
                    current_row = row
                elif row['nlm_unique_id'] == current_row['nlm_unique_id'] and \
                    row['holdings_format'] == current_row['holdings_format'] and \
                    row['begin_year'] > current_row['end_year']:
                    left_effective_date=row['end_year'] - ((row['embargo_period'] / 12))
                    right_effective_date=current_row['end_year'] - ((current_row['embargo_period'] / 12))
                    if current_row['embargo_period'] != 0 and current_row['embargo_period'] is not None and right_effective_date > left_effective_date:
                        current_row['embargo_period'] = 0
                    output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)

                elif row['nlm_unique_id'] == current_row['nlm_unique_id'] and \
                    row['holdings_format'] == current_row['holdings_format'] and \
                    row['begin_year'] <= current_row['end_year'] and \
                    row['embargo_period'] != current_row['embargo_period']:
                    left_effective_date=row['end_year'] - ((row['embargo_period'] / 12))
                    right_effective_date=current_row['end_year'] - ((current_row['embargo_period'] / 12))
                    if left_effective_date > right_effective_date:
                        current_row['end_year'] = row['end_year']
                        current_row['embargo_period'] = row['embargo_period']
                    elif left_effective_date == right_effective_date and current_row['embargo_period'] > row['embargo_period']:
                        left_embargo_period = row['embargo_period']
                        right_embargo_period  = current_row['embargo_period']

                        if left_embargo_period < right_embargo_period:
                            current_row['end_year'] = row['end_year']
                            current_row['embargo_period'] = row['embargo_period']

                elif row['nlm_unique_id'] == current_row['nlm_unique_id'] and \
                    row['holdings_format'] == current_row['holdings_format'] and \
                    row['begin_year'] <= current_row['end_year'] and \
                    row['embargo_period'] == current_row['embargo_period']:
                    # Extend the current range if overlapping
                    current_row['end_year'] = max(current_row['end_year'], row['end_year'])

                     # elif elif row['nlm_unique_id'] == current_row['nlm_unique_id'] and
                     #     row['holdings_format'] == current_row['holdings_format'] and
                     #     row['begin_year'] > current_row['end_year']:
                     #     #row['embargo_period']) == current_row['embargo_period']:
                     #
                     #     current_row['end_year'] = max(current_row['end_year'], row['end_year'])
            else:
                current_row = row
                #output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)
    # Append the last range row if it exists
    if current_row is not None and current_row['record_type'] == 'RANGE':
        output_df=pd.concat([output_df, pd.DataFrame([current_row])], ignore_index=True)

    return output_df


# # Load the dataset
# new_file_path = '/path/to/your/file.csv'  # Replace with your file path
# new_df = pd.read_csv(new_file_path)



def apply_currently_received(updated_df):
    updated_HOLDING_df = updated_df.copy()
    updated_HOLDING_df = updated_df[updated_df['record_type'] == "HOLDING"]
    updated_RANGE_df = updated_df.copy()
    updated_RANGE_df = updated_df[updated_df['record_type'] == "RANGE"]

    for index, row in updated_HOLDING_df.iterrows():
        nlm_left = row['nlm_unique_id']
        holdings_format_left = row['holdings_format']
        action_left = row['action']
        rows_df = updated_RANGE_df[(updated_RANGE_df['nlm_unique_id'] == nlm_left) & (
            updated_RANGE_df['holdings_format'] == holdings_format_left) & (updated_RANGE_df['action'] == action_left)]
        currently_received_list = rows_df['currently_received'].tolist()
        assignment_value = "No"
        for c_r in currently_received_list:
            if c_r == "Yes":
                assignment_value = "Yes"
                break
        if assignment_value != "Yes":
            assignment_value = "No"
        updated_df.loc[(updated_df['nlm_unique_id'] == nlm_left) & (updated_df['holdings_format'] == holdings_format_left) & (
            updated_df['action'] == action_left) & (updated_df['record_type'] == "HOLDING"), 'currently_received'] = assignment_value

    return updated_df


def merge_intervals_optimized(df):
    # Sort the DataFrame
    df.sort_values(by=['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year'], ascending=[True, True, False, True, True, True], inplace=True)
    # Using 10000 to represent 'indefinite'
    df['end_year'].fillna(10000, inplace=True)

    output_df = pd.DataFrame(columns=df.columns)
    current_row = None

    for _, row in df.iterrows():
        if row['record_type'] == 'HOLDING':
            output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)
        elif row['record_type'] == 'RANGE':


            if current_row is not None:
                # if row['embargo_period']) == current_row['embargo_period'] and
                if row['nlm_unique_id'] != current_row['nlm_unique_id'] or \
                    row['holdings_format'] != current_row['holdings_format']:
                    output_df = pd.concat([output_df, pd.DataFrame([current_row])], ignore_index=True)
                    current_row = row
                elif row['nlm_unique_id'] == current_row['nlm_unique_id'] and \
                    row['holdings_format'] == current_row['holdings_format'] and \
                    row['begin_year'] > current_row['end_year']:
                    left_effective_date=row['end_year'] - ((row['embargo_period'] / 12))
                    right_effective_date=current_row['end_year'] - ((current_row['embargo_period'] / 12))
                    if current_row['embargo_period'] != 0 and current_row['embargo_period'] is not None and right_effective_date > left_effective_date:
                        current_row['embargo_period'] = 0
                    output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)

                elif row['nlm_unique_id'] == current_row['nlm_unique_id'] and \
                    row['holdings_format'] == current_row['holdings_format'] and \
                    row['begin_year'] <= current_row['end_year'] and \
                    row['embargo_period'] != current_row['embargo_period']:
                    left_effective_date=row['end_year'] - ((row['embargo_period'] / 12))
                    right_effective_date=current_row['end_year'] - ((current_row['embargo_period'] / 12))
                    if left_effective_date > right_effective_date:
                        current_row['end_year'] = row['end_year']
                        current_row['embargo_period'] = row['embargo_period']
                    elif left_effective_date == right_effective_date and current_row['embargo_period'] > row['embargo_period']:
                        left_embargo_period = row['embargo_period']
                        right_embargo_period  = current_row['embargo_period']

                        if left_embargo_period < right_embargo_period:
                            current_row['end_year'] = row['end_year']
                            current_row['embargo_period'] = row['embargo_period']

                elif row['nlm_unique_id'] == current_row['nlm_unique_id'] and \
                    row['holdings_format'] == current_row['holdings_format'] and \
                    row['begin_year'] <= current_row['end_year'] and \
                    row['embargo_period'] == current_row['embargo_period']:
                    # Extend the current range if overlapping
                    current_row['end_year'] = max(current_row['end_year'], row['end_year'])

                     # elif elif row['nlm_unique_id'] == current_row['nlm_unique_id'] and
                     #     row['holdings_format'] == current_row['holdings_format'] and
                     #     row['begin_year'] > current_row['end_year']:
                     #     #row['embargo_period']) == current_row['embargo_period']:
                     #
                     #     current_row['end_year'] = max(current_row['end_year'], row['end_year'])
            else:
                current_row = row
                #output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)
    # Append the last range row if it exists
    if current_row is not None and current_row['record_type'] == 'RANGE':
        output_df=pd.concat([output_df, pd.DataFrame([current_row])], ignore_index=True)

    return output_df



# # Load the dataset
# new_file_path = '/path/to/your/file.csv'  # Replace with your file path
# new_df = pd.read_csv(new_file_path)


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
    try:
        deleted_alma_df = deleted_alma_df[deleted_alma_df['Bibliographic Lifecycle'] == 'Deleted']
    except:
        deleted_alma_df = deleted_alma_df[deleted_alma_df['Bibliographic Lifecycle'] == 'Deleted']
    #
    current_alma_df = alma_nlm_merge_df.copy()

    current_alma_df = current_alma_df[~current_alma_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(deleted_alma_df.set_index(['nlm_unique_id', 'holdings_format']).index)]

    deleted_alma_df = deleted_alma_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'begin_volume', 'end_year', 'end_volume'], ascending = [True, True, True, True, True, True], na_position = 'first')

    current_alma_df = current_alma_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'begin_volume', 'end_year', 'end_volume'], ascending = [True, True, True, True, True, True], na_position = 'first')


    existing_docline_df_for_compare = existing_docline_df.copy()
    existing_docline_df_for_compare['nlm_unique_id'] = existing_docline_df_for_compare['nlm_unique_id'].apply(lambda x: x.replace("NLM_", ""))
    # print(existing_docline_df_for_compare)


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



    current_alma_df['end_year'] = current_alma_df['end_year'].apply(lambda x: str(x).replace("10000", ""))
    current_alma_df['end_year'] = pd.to_numeric(current_alma_df['end_year'], errors='coerce')
    current_alma_df['end_year'] = current_alma_df['end_year'].astype('Int64')
    #current_alma_df['end_year'] = current_alma_df['end_year'].replace(0, np.nan)


    current_alma_df['begin_year'] = current_alma_df['begin_year'].apply(lambda x: str(x).replace("10000", ""))
    current_alma_df['begin_year'] = pd.to_numeric(current_alma_df['begin_year'], errors='coerce')
    current_alma_df['begin_year'] = current_alma_df['begin_year'].astype('Int64')
    #current_alma_df['begin_year'] = current_alma_df['begin_year'].replace(0, np.nan)

    # current_alma_df['end_volume'] = current_alma_df['end_volume'].apply(lambda x: str(x).replace("10000", ""))
    # current_alma_df['end_volume'] = pd.to_numeric(current_alma_df['end_volume'], errors='coerce')
    # current_alma_df['end_volume'] = current_alma_df['end_volume'].astype('Int64')
    # current_alma_df['end_volume'] = current_alma_df['end_volume'].replace(0, np.nan)
    #
    # current_alma_df['begin_volume'] = current_alma_df['begin_volume'].apply(lambda x: str(x).replace("10000", ""))
    # current_alma_df['begin_volume'] = pd.to_numeric(current_alma_df['begin_volume'], errors='coerce')
    # current_alma_df['begin_volume'] = current_alma_df['begin_volume'].astype('Int64')
    # current_alma_df['begin_volume'] = current_alma_df['begin_volume'].replace(0, np.nan)

    current_alma_df['embargo_period'] = current_alma_df['embargo_period'].apply(lambda x: str(x).replace("10000", ""))
    current_alma_df['embargo_period'] = pd.to_numeric(current_alma_df['embargo_period'], errors='coerce')
    current_alma_df['embargo_period'] = current_alma_df['embargo_period'].astype('Int64')
    #current_alma_df['embargo_period'] = current_alma_df['embargo_period'].replace(0, np.nan)



    current_alma_df = merge_intervals_optimized(current_alma_df.copy())

    current_alma_df.sort_values(by=['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year'], inplace=True)

    current_alma_df.loc[current_alma_df['end_year'] == 10000, 'currently_received'] = "Yes"


    current_alma_df = apply_currently_received(current_alma_df)

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


    try:
        current_alma_df = current_alma_df.drop('index', axis=1)
    except:
        drop_index = False
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

    add_df.to_csv("Processing/Add Before Any Other Processing and After reconciling Against Delete.csv", index=False)

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
    # all_columns.remove('Bibliographic Lifecycle')
    all_columns.remove('begin_volume')
    all_columns.remove('end_volume')
    all_columns.remove('action')
    all_columns.remove('ignore_warnings')



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


    # current_alma_df[aggregate_columns] = pd.to_numeric(current_alma_df[aggregate_columns], errors='coerce')
    matched_base_df_alma_side[aggregate_columns] = matched_base_df_alma_side[aggregate_columns].astype('Int64')
    matched_base_df_alma_side[aggregate_columns] = matched_base_df_alma_side[aggregate_columns].replace(0, np.nan)

    # existing_docline_df_for_compare[aggregate_columns] = pd.to_numeric(existing_docline_df_for_compare[aggregate_columns], errors='coerce')
    existing_docline_df_for_compare[aggregate_columns] = existing_docline_df_for_compare[aggregate_columns].astype('Int64')
    existing_docline_df_for_compare[aggregate_columns] = existing_docline_df_for_compare[aggregate_columns].replace(0, np.nan)
    #
    # current_alma_df[aggregate_columns] = current_alma_df[aggregate_columns].astype('float')
    # existing_docline_df_for_compare[aggregate_columns] = existing_docline_df_for_compare[aggregate_columns].astype('float')

    matched_base_df_alma_side[aggregate_columns] = matched_base_df_alma_side[aggregate_columns].astype('Int64')
    existing_docline_df_for_compare[aggregate_columns] = existing_docline_df_for_compare[aggregate_columns].astype('Int64')



    # note the group by columns being those fields in the holding row,
    # and the aggregate columns being those things in range, that are being rolled up
    existing_docline_for_compare_agg_df = existing_docline_df_for_compare.copy()
    matched_nlm_alma_df_for_compare_agg = matched_base_df_alma_side.copy()



    existing_docline_for_compare_agg_df = existing_docline_for_compare_agg_df.groupby(group_by_columns, dropna=False, as_index=False).agg({aggregate_columns[0]: lambda x: '; '.join(set(x.astype(str))), aggregate_columns[1]: lambda x: '; '.join(set(x.astype(str)))})
    matched_nlm_alma_df_for_compare_agg = matched_nlm_alma_df_for_compare_agg.groupby(group_by_columns, dropna=False, as_index=False).agg({aggregate_columns[0]: lambda x: '; '.join(set(x.astype(str))), aggregate_columns[1]: lambda x: '; '.join(set(x.astype(str)))})



    # in the rolled up range data, remove the appearance of "<NA>" in the stringified list
    matched_nlm_alma_df_for_compare_agg[['begin_year', 'end_year']] = matched_nlm_alma_df_for_compare_agg[['begin_year', 'end_year']].applymap(lambda x: removeNA(x))



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




    #do I need this
    #different_ranges_alma_output_df.to_excel('Processing/Different Ranges Alma - After -Add- Action Entered.xlsx', index=False)
    #in_docline_only_preserve_df['action'] = 'ADD'


    # try:
    #     merged_updated_df = merged_updated_df.drop(columns=['record_type_x', 'action_x'])
    # except:
    #     print("no dupliate action and record type columns")
    add_df['action'] = 'ADD'
    deleted_output_df['action'] = 'DELETE'

    # do I need this?
    full_match_output_df['action'] = 'ADD'
    different_ranges_alma_output_df['action'] = 'ADD'
    different_ranges_docline_output_df['action']= 'DELETE'

    different_ranges_alma_output_df = different_ranges_alma_output_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True], na_position = 'first')
    different_ranges_docline_output_df = different_ranges_docline_output_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True], na_position = 'first')



    no_dates_df = different_ranges_alma_output_df.copy()

    no_dates_df = different_ranges_alma_output_df[different_ranges_alma_output_df['begin_year'].isna()]
    #different_ranges_alma_output_df = different_ranges_alma_output_df[(~different_ranges_alma_output_df['begin_year'].isna()) & (different_ranges_alma_output_df['record_type'] == 'RANGE')]


    no_dates_df.to_csv("Output/No Dates in Update Table.xlsx")
    different_ranges_alma_output_df = different_ranges_alma_output_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True, True], na_position = 'first')
    #filtered_new_df = different_ranges_alma_output_df[['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year']]#.dropna(subset=['begin_year'])
    different_ranges_alma_output_df.to_csv('Processing/Different Ranges Alma Before Compression.csv', index=False)
    # updated_df = merge_intervals_optimized(different_ranges_alma_output_df.copy())
    #
    # updated_df.sort_values(by=['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year'], inplace=True)
    # ## Merge the original dataframe with the merged intervals
    # ##updated_df = different_ranges_alma_output_df.drop(columns=['begin_year', 'end_year']).merge(merged_new_df_optimized, on=['nlm_unique_id', 'holdings_format', 'action', 'record_type'], how='left')
    #
    # # Optional: Remove duplicate rows based on the compound key
    # ##updated_df.drop_duplicates(subset=['nlm_unique_id', 'holdings_format', 'action', 'record_type'], inplace=True)
    # updated_df.loc[updated_df['end_year'] == 10000, 'currently_received'] = "Yes"
    #
    #
    # updated_df = apply_currently_received(updated_df)

    ## updated_df = updated_df[columns]
    merged_updated_df = pd.concat([different_ranges_alma_output_df, different_ranges_docline_output_df])

    merged_updated_df = merged_updated_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True, True], na_position = 'first')



    # merged_updated_df['begin_volume'] = None
    # merged_updated_df['end_volume'] = None
    merged_updated_df['ignore_warnings'] = "Yes"
    merged_updated_df = merged_updated_df[((~merged_updated_df['begin_year'].isna()) | (~merged_updated_df['begin_volume'].isna()) | (merged_updated_df['record_type'] == "HOLDING"))]
    #merged_updated_df = merged_updated_df[(merged_updated_df['nlm_unique_id'] != "NLM_0052457") & (merged_updated_df['nlm_unique_id'] != "NLM_0154720") & (merged_updated_df['nlm_unique_id'] != "NLM_0255562") & (merged_updated_df['nlm_unique_id'] != "NLM_0330471") & (merged_updated_df['nlm_unique_id'] != "NLM_0372435")  & (merged_updated_df['nlm_unique_id'] != "NLM_0326264")]









    add_df.to_csv("Output/Unmerged_Add.csv", index=False)
    no_dates_add_df = add_df.copy()


    no_dates_add_df = add_df[add_df['begin_year'].isna()]
    #different_ranges_alma_output_df = different_ranges_alma_output_df[(~different_ranges_alma_output_df['begin_year'].isna()) & (different_ranges_alma_output_df['record_type'] == 'RANGE')]

    no_dates_add_df.to_csv("Output/No Dates in Add Table.xlsx")
    add_df = add_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True, True], na_position = 'first')
    #filtered_new_df = add_df[['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year']]#.dropna(subset=['begin_year'])
    add_df.to_csv("Processing/Add DF Before Compression.csv", index=False)

    # updated_add_df = merge_intervals_optimized(add_df.copy())
    #
    # ## Merge the original dataframe with the merged intervals
    # ##updated_add_df = add_df.drop(columns=['begin_year', 'end_year']).merge(merged_new_df_optimized, on=['nlm_unique_id', 'holdings_format', 'action', 'record_type'], how='left')
    #
    # ## Optional: Remove duplicate rows based on the compound key
    # ##updated_add_df.drop_duplicates(subset=['nlm_unique_id', 'holdings_format', 'action', 'record_type'], inplace=True)
    #
    #
    # updated_add_df = updated_add_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True, True], na_position = 'first')
    #
    #
    #
    # ##merged_updated_df = pd.concat([updated_df, different_ranges_docline_output_df])
    # updated_add_df.loc[updated_add_df['end_year'] == 10000, 'currently_received'] = "Yes"
    #
    #
    # updated_df = apply_currently_received(updated_df)
    # ##updated_add_df = updated_add_df.reset_index()
    # updated_add_df = updated_add_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period','begin_year', 'end_year'], ascending = [True, True, False, True, True, True, True], na_position = 'first')


    # updated_add_df = updated_add_df[columns_add]

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


    # print("add_df length")
    # print(len(add_df))
    # print(len(pd.unique(add_df['nlm_unique_id'])))

    add_df = add_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, True, True, True, True], na_position = 'first')

    deleted_output_df = deleted_output_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, True, True, True, True], na_position = 'first')
    #deleted_output_df.to_excel('Processing/Deleted DF After Sort.xlsx', index=False)
    full_match_output_df = full_match_output_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'embargo_period', 'record_type', 'begin_year', 'end_year'], ascending = [True, True, True, True, True, True])
    #different_ranges_alma_output_df = different_ranges_alma_output_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'end_year'], ascending = [True, True, True, True], na_position = 'first')
    #different_ranges_alma_output_df.to_excel('Processing/Add Alma After Sort.xlsx', index=False)
    #different_ranges_docline_output_df = different_ranges_docline_output_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'end_year'], ascending = [True, True, True, True], na_position = 'first')
    in_docline_only_preserve_df = in_docline_only_preserve_df.sort_values(by = ['serial_title', 'nlm_unique_id', 'begin_year', 'end_year'], ascending = [True, True, True, True], na_position = 'first')
    # print("\n\nbefore reset index\n\n")
    # print(different_ranges_alma_output_df)
    #different_ranges_alma_output_df = different_ranges_alma_output_df.set_index('index')#reset_index()
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
        merged_updated_df = merged_updated_df.drop('Lifecycle', axis=1)

    except:
        print("Lifecycle column already removed")
    try:
        merged_updated_df = merged_updated_df.drop('level_0', axis=1)
    except:
        print("level 0 already removed")
    # print("\n\nafter remove level 0 and before remove bibliographic lifecycle\n\n")
    #
    # print(different_ranges_alma_output_df)
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

    # print("\n\nafter remove bibliographic lifecycle\n\n")
    #
    # print(different_ranges_alma_output_df)


    # try:
    #     in_docline_only_preserve_df = in_docline_only_preserve_df.drop('index', axis=1)
    #
    # except:

    try:
        add_df = add_df.drop('level_0', axis=1)
    except:
        print("level 0 already removed")
    try:
        add_df = add_df.drop('index', axis=1)
    except:
        print("index already removed")
    # print("\n\nafter remove level 0 and before remove bibliographic lifecycle\n\n")
    #
    # print(different_ranges_alma_output_df)
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

    # print("\n\nafter remove bibliographic lifecycle\n\n")
    #
    # print(different_ranges_alma_output_df)


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
    different_ranges_docline_output_df.to_csv('Output/Different Ranges Docline Final.csv', index=False)
    different_ranges_alma_output_df.to_csv('Output/Different Ranges Alma Final.csv', index=False)
    in_docline_only_preserve_df = in_docline_only_preserve_df.reset_index()
    counts_df = pd.concat([counts_df, pd.DataFrame({'Set': 'In Docline Only Keep', 'Number of Rows': len(in_docline_only_preserve_df), 'Number of NLM Unique IDs': len(pd.unique(in_docline_only_preserve_df['nlm_unique_id']))}, index=[0])])

    try:
        in_docline_only_preserve_df = in_docline_only_preserve_df.drop('index', axis=1)

    except:
        print('index already removed')
    in_docline_only_preserve_df.to_csv('Output/In Docline Only Preserve Final.csv', index=False)

    counts_df.to_excel('Output/Counts.xlsx', index=False)


startTime = datetime.now()

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






merge(alma_nlm_merge_df, existing_docline_df)

print("Execution time:\t")
print(datetime.now() - startTime)
#
# matched_nlm_alma_df_for_compare_agg = pd.read_excel('Processing/matched_nlm_alma_df_for_compare_agg.xlsx', engine="openpyxl")
#
# existing_docline_for_compare_agg_df = pd.read_excel("Processing/existing_docline_for_compare_agg_df.xlsx", engine='openpyxl')
# alma_nlm_merge_df = pd.read_csv("Processing/Alma Docline output.csv", engine='python')
# #
# existing_docline_df_for_compare = pd.read_excel('Processing/Existing Docline for Compare.xlsx', engine="openpyxl")
# all_columns = existing_docline_df_for_compare.columns
#
#
# aggregate_columns = []
# group_by_columns = []
# for column in all_columns:
#     if column in ['begin_year', 'end_year']:
#         aggregate_columns.append(column)
#     elif column == 'Bibliographic Lifecycle' or column == 'last_modified' or column == 'issns' or column == 'begin_volume' or column == 'end_volume':
#         continue
#     else:
#         group_by_columns.append(column)
#
#
# all_columns = all_columns.tolist()
#
# all_columns.remove('last_modified')
# all_columns.remove('issns')
# # all_columns.remove('Bibliographic Lifecycle')
# all_columns.remove('begin_volume')
# all_columns.remove('end_volume')
#
#
#
#
#
# different_ranges_alma_compare_df_agg = matched_nlm_alma_df_for_compare_agg.copy()
# different_ranges_alma_compare_df_agg = different_ranges_alma_compare_df_agg[(different_ranges_alma_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index.isin(existing_docline_for_compare_agg_df.set_index(['nlm_unique_id', 'holdings_format']).index)) & (~different_ranges_alma_compare_df_agg.set_index(all_columns).index.isin(existing_docline_for_compare_agg_df.set_index(all_columns).index))]
# # ditto to above, but among the current Docline set compared to Alma (other direction)
# different_ranges_docline_compare_df_agg = existing_docline_for_compare_agg_df.copy()
# #                                     different_ranges_alma_compare_df_agg[(different_ranges_alma_compare_df_agg.nlm_unique_id.isin(existing_docline_for_compare_agg_df.nlm_unique_id)) & (~different_ranges_alma_compare_df_agg.set_index(all_columns).index.isin(existing_docline_for_compare_agg_df.set_index(all_columns).index))]
# # different_ranges_docline_compare_df_agg = existing_docline_for_compare_agg_df[(existing_docline_for_compare_agg_df.nlm_unique_id.isin(matched_nlm_alma_df_for_compare_agg.nlm_unique_id)) & (~existing_docline_for_compare_agg_df.set_index(all_columns).index.isin(matched_nlm_alma_df_for_compare_agg.set_index(all_columns).index)) & (existing_docline_for_compare_agg_df['holdings_format'] == matched_nlm_alma_df_for_compare_agg['holdings_format'])]
# different_ranges_docline_compare_df_agg = different_ranges_docline_compare_df_agg[(different_ranges_docline_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index.isin(matched_nlm_alma_df_for_compare_agg.set_index(['nlm_unique_id', 'holdings_format']).index)) & (~different_ranges_docline_compare_df_agg.set_index(all_columns).index.isin(matched_nlm_alma_df_for_compare_agg.set_index(all_columns).index))]
# # different_ranges_docline_compare_df_agg = different_ranges_docline_compare_df_agg.drop('index', axis=1)
#
# # print(len(full_match_output_df['nlm_unique_id'].value_counts()))
# different_ranges_alma_output_df = alma_nlm_merge_df.copy()
# different_ranges_alma_output_df = different_ranges_alma_output_df[different_ranges_alma_output_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(different_ranges_alma_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index)]
# different_ranges_alma_output_df = different_ranges_alma_output_df.reset_index()
# # counts_df = pd.concat([counts_df, pd.DataFrame({'Set': 'Alma Different Ranges', 'Number of Rows': len(different_ranges_alma_output_df), 'Number of NLM Unique IDs': len(pd.unique(different_ranges_alma_output_df['nlm_unique_id']))}, index=[0])])
# different_ranges_alma_output_df = different_ranges_alma_output_df.reset_index()
# different_ranges_docline_output_df = existing_docline_df_for_compare.copy()
# different_ranges_docline_output_df = existing_docline_df_for_compare[existing_docline_df_for_compare.set_index(['nlm_unique_id', 'holdings_format']).index.isin(different_ranges_docline_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index)]
#
# different_ranges_alma_output_df['nlm_unique_id'] = different_ranges_alma_output_df['nlm_unique_id'].apply(lambda x: "NLM_" + x)
# different_ranges_docline_output_df['nlm_unique_id'] = different_ranges_docline_output_df['nlm_unique_id'].apply(lambda x: "NLM_" + x)
#
# different_ranges_alma_output_df['action'] = 'ADD'
# different_ranges_docline_output_df['action']= 'DELETE'
# different_ranges_alma_output_df = different_ranges_alma_output_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True], na_position = 'first')
# different_ranges_docline_output_df = different_ranges_docline_output_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True], na_position = 'first')
# no_dates_df = different_ranges_alma_output_df.copy()
# no_dates_df = different_ranges_alma_output_df[different_ranges_alma_output_df['begin_year'].isna()]
# # different_ranges_alma_output_df = different_ranges_alma_output_df[(~different_ranges_alma_output_df['begin_year'].isna()) & (different_ranges_alma_output_df['record_type'] == 'RANGE')]
# no_dates_df.to_csv("Output/No Dates in Update Table.xlsx")
# different_ranges_alma_output_df = different_ranges_alma_output_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True, True], na_position = 'first')
# # filtered_new_df = different_ranges_alma_output_df[['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year']]#.dropna(subset=['begin_year'])
# different_ranges_alma_output_df.to_csv('Processing/Different Ranges Alma Before Compression.csv', index=False)
# #updated_df = merge_intervals_optimized(different_ranges_alma_output_df.copy())
# updated_df = different_ranges_alma_output_df.copy()
# #updated_df['end_year'].fillna(10000, inplace=True)
#
# updated_df.sort_values(by=['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year'], ascending=[True, True, False, True, True, True], inplace=True)
#
# # Merge the original dataframe with the merged intervals
# # updated_df = different_ranges_alma_output_df.drop(columns=['begin_year', 'end_year']).merge(merged_new_df_optimized, on=['nlm_unique_id', 'holdings_format', 'action', 'record_type'], how='left')
# # Optional: Remove duplicate rows based on the compound key
# # updated_df.drop_duplicates(subset=['nlm_unique_id', 'holdings_format', 'action', 'record_type'], inplace=True)
# updated_df.loc[updated_df['end_year'] == 10000, 'currently_received'] = "Yes"
#
# updated_df = apply_currently_received(updated_df)
# # if index != 0:
# # if row['record_type'] == "RANGE" and updated_df.loc[index - 1, 'record_type'] == "HOLDING":
# # updated_df.loc[index - 1, 'currently_received'] = row['currently_received']
# # updated_df = updated_df[columns]
# merged_updated_df = pd.concat([updated_df, different_ranges_docline_output_df])
# merged_updated_df = merged_updated_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True, True], na_position = 'first')
# merged_updated_df['end_year'] = merged_updated_df['end_year'].apply(lambda x: str(x).replace("10000", ""))
# merged_updated_df['end_year'] = pd.to_numeric(merged_updated_df['end_year'], errors='coerce')
# merged_updated_df['end_year'] = merged_updated_df['end_year'].astype('Int64')
# merged_updated_df['end_year'] = merged_updated_df['end_year'].replace(0, np.nan)
#
# merged_updated_df['begin_year'] = merged_updated_df['begin_year'].apply(lambda x: str(x).replace("10000", ""))
# merged_updated_df['begin_year'] = pd.to_numeric(merged_updated_df['begin_year'], errors='coerce')
# merged_updated_df['begin_year'] = merged_updated_df['begin_year'].astype('Int64')
# merged_updated_df['begin_year'] = merged_updated_df['begin_year'].replace(0, np.nan)
#
# merged_updated_df['end_volume'] = merged_updated_df['end_volume'].apply(lambda x: str(x).replace("10000", ""))
# merged_updated_df['end_volume'] = pd.to_numeric(merged_updated_df['end_volume'], errors='coerce')
# merged_updated_df['end_volume'] = merged_updated_df['end_volume'].astype('Int64')
# merged_updated_df['end_volume'] = merged_updated_df['end_volume'].replace(0, np.nan)
#
# merged_updated_df['begin_volume'] = merged_updated_df['begin_volume'].apply(lambda x: str(x).replace("10000", ""))
# merged_updated_df['begin_volume'] = pd.to_numeric(merged_updated_df['begin_volume'], errors='coerce')
# merged_updated_df['begin_volume'] = merged_updated_df['begin_volume'].astype('Int64')
# merged_updated_df['begin_volume'] = merged_updated_df['begin_volume'].replace(0, np.nan)
#
# # merged_updated_df['begin_volume'] = None
# # merged_updated_df['end_volume'] = None
# merged_updated_df['ignore_warnings'] = "Yes"
# merged_updated_df = merged_updated_df[((~merged_updated_df['begin_year'].isna()) | (~merged_updated_df['begin_volume'].isna()) | (merged_updated_df['record_type'] == "HOLDING"))]
# # merged_updated_df = merged_updated_df[(merged_updated_df['nlm_unique_id'] != "NLM_0052457") & (merged_updated_df['nlm_unique_id'] != "NLM_0154720") & (merged_updated_df['nlm_unique_id'] != "NLM_0255562") & (merged_updated_df['nlm_unique_id'] != "NLM_0330471") & (merged_updated_df['nlm_unique_id'] != "NLM_0372435")  & (merged_updated_df['nlm_unique_id'] != "NLM_0326264")]
# try:
#     merged_updated_df = merged_updated_df.drop('level_0', axis=1)
# except:
#     print("level 0 already removed")
# # print("\n\nafter remove level 0 and before remove bibliographic lifecycle\n\n")
# #
# # print(different_ranges_alma_output_df)
# try:
#     merged_updated_df = merged_updated_df.drop('Lifecycle', axis=1)
#
# except:
#     print("Lifecycle column already removed")
# try:
#     merged_updated_df = merged_updated_df.drop('Bibliographic Lifecycle', axis=1)
#
# except:
#     print("Bibliographic Lifecycle column already removed")
#
# # print("\n\nafter remove bibliographic lifecycle\n\n")
# #
# # print(different_ranges_alma_output_df)
# try:
#     merged_updated_df = merged_updated_df.drop('index', axis=1)
#
# except:
#     "index column already removed"
# try:
#     merged_updated_df = merged_updated_df.drop('libid', axis=1)
#
# except:
#     "libid column already removed"
#
#
# merged_updated_df.to_csv('Output/Update Final.csv', index=False)
