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
import secrets_local
import pandas as pd
import numpy as np
import time
import os
import sys
sys.path.append('secrets_local/')


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
    df.sort_values(by=['nlm_unique_id', 'holdings_format', 'action', 'record_type',
                   'begin_year', 'end_year'], ascending=[True, True, False, True, True, True], inplace=True)
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


matched_nlm_alma_df_for_compare_agg = pd.read_excel('Processing/matched_nlm_alma_df_for_compare_agg.xlsx', engine="openpyxl")

existing_docline_for_compare_agg_df = pd.read_excel("Processing/existing_docline_for_compare_agg_df.xlsx", engine='openpyxl')
alma_nlm_merge_df = pd.read_csv("Processing/Alma Docline output.csv", engine='python')
#
existing_docline_df_for_compare = pd.read_excel('Processing/Existing Docline for Compare.xlsx', engine="openpyxl")
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





different_ranges_alma_compare_df_agg = matched_nlm_alma_df_for_compare_agg.copy()
different_ranges_alma_compare_df_agg = different_ranges_alma_compare_df_agg[(different_ranges_alma_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index.isin(existing_docline_for_compare_agg_df.set_index(['nlm_unique_id', 'holdings_format']).index)) & (~different_ranges_alma_compare_df_agg.set_index(all_columns).index.isin(existing_docline_for_compare_agg_df.set_index(all_columns).index))]
# ditto to above, but among the current Docline set compared to Alma (other direction)
different_ranges_docline_compare_df_agg = existing_docline_for_compare_agg_df.copy()
#                                     different_ranges_alma_compare_df_agg[(different_ranges_alma_compare_df_agg.nlm_unique_id.isin(existing_docline_for_compare_agg_df.nlm_unique_id)) & (~different_ranges_alma_compare_df_agg.set_index(all_columns).index.isin(existing_docline_for_compare_agg_df.set_index(all_columns).index))]
# different_ranges_docline_compare_df_agg = existing_docline_for_compare_agg_df[(existing_docline_for_compare_agg_df.nlm_unique_id.isin(matched_nlm_alma_df_for_compare_agg.nlm_unique_id)) & (~existing_docline_for_compare_agg_df.set_index(all_columns).index.isin(matched_nlm_alma_df_for_compare_agg.set_index(all_columns).index)) & (existing_docline_for_compare_agg_df['holdings_format'] == matched_nlm_alma_df_for_compare_agg['holdings_format'])]
different_ranges_docline_compare_df_agg = different_ranges_docline_compare_df_agg[(different_ranges_docline_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index.isin(matched_nlm_alma_df_for_compare_agg.set_index(['nlm_unique_id', 'holdings_format']).index)) & (~different_ranges_docline_compare_df_agg.set_index(all_columns).index.isin(matched_nlm_alma_df_for_compare_agg.set_index(all_columns).index))]
# different_ranges_docline_compare_df_agg = different_ranges_docline_compare_df_agg.drop('index', axis=1)

# print(len(full_match_output_df['nlm_unique_id'].value_counts()))
different_ranges_alma_output_df = alma_nlm_merge_df.copy()
different_ranges_alma_output_df = different_ranges_alma_output_df[different_ranges_alma_output_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(different_ranges_alma_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index)]
different_ranges_alma_output_df = different_ranges_alma_output_df.reset_index()
# counts_df = pd.concat([counts_df, pd.DataFrame({'Set': 'Alma Different Ranges', 'Number of Rows': len(different_ranges_alma_output_df), 'Number of NLM Unique IDs': len(pd.unique(different_ranges_alma_output_df['nlm_unique_id']))}, index=[0])])
different_ranges_alma_output_df = different_ranges_alma_output_df.reset_index()
different_ranges_docline_output_df = existing_docline_df_for_compare.copy()
different_ranges_docline_output_df = existing_docline_df_for_compare[existing_docline_df_for_compare.set_index(['nlm_unique_id', 'holdings_format']).index.isin(different_ranges_docline_compare_df_agg.set_index(['nlm_unique_id', 'holdings_format']).index)]

different_ranges_alma_output_df['nlm_unique_id'] = different_ranges_alma_output_df['nlm_unique_id'].apply(lambda x: "NLM_" + x)
different_ranges_docline_output_df['nlm_unique_id'] = different_ranges_docline_output_df['nlm_unique_id'].apply(lambda x: "NLM_" + x)

different_ranges_alma_output_df['action'] = 'ADD'
different_ranges_docline_output_df['action']= 'DELETE'
different_ranges_alma_output_df = different_ranges_alma_output_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True], na_position = 'first')
different_ranges_docline_output_df = different_ranges_docline_output_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True], na_position = 'first')
no_dates_df = different_ranges_alma_output_df.copy()
no_dates_df = different_ranges_alma_output_df[different_ranges_alma_output_df['begin_year'].isna()]
# different_ranges_alma_output_df = different_ranges_alma_output_df[(~different_ranges_alma_output_df['begin_year'].isna()) & (different_ranges_alma_output_df['record_type'] == 'RANGE')]
no_dates_df.to_csv("Output/No Dates in Update Table.xlsx")
different_ranges_alma_output_df = different_ranges_alma_output_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True, True], na_position = 'first')
# filtered_new_df = different_ranges_alma_output_df[['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year']]#.dropna(subset=['begin_year'])
different_ranges_alma_output_df.to_csv('Processing/Different Ranges Alma Before Compression.csv', index=False)
updated_df = merge_intervals_optimized(different_ranges_alma_output_df.copy())
updated_df.sort_values(by=['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year'], inplace=True)
# Merge the original dataframe with the merged intervals
# updated_df = different_ranges_alma_output_df.drop(columns=['begin_year', 'end_year']).merge(merged_new_df_optimized, on=['nlm_unique_id', 'holdings_format', 'action', 'record_type'], how='left')
# Optional: Remove duplicate rows based on the compound key
# updated_df.drop_duplicates(subset=['nlm_unique_id', 'holdings_format', 'action', 'record_type'], inplace=True)
updated_df.loc[updated_df['end_year'] == 10000, 'currently_received'] = "Yes"

updated_df = apply_currently_received(updated_df)
# if index != 0:
# if row['record_type'] == "RANGE" and updated_df.loc[index - 1, 'record_type'] == "HOLDING":
# updated_df.loc[index - 1, 'currently_received'] = row['currently_received']
# updated_df = updated_df[columns]
merged_updated_df = pd.concat([updated_df, different_ranges_docline_output_df])
merged_updated_df = merged_updated_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'embargo_period', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True, True], na_position = 'first')
merged_updated_df['end_year'] = merged_updated_df['end_year'].apply(lambda x: str(x).replace("10000", ""))
merged_updated_df['end_year'] = pd.to_numeric(merged_updated_df['end_year'], errors='coerce')
merged_updated_df['end_year'] = merged_updated_df['end_year'].astype('Int64')
merged_updated_df['end_year'] = merged_updated_df['end_year'].replace(0, np.nan)

merged_updated_df['begin_year'] = merged_updated_df['begin_year'].apply(lambda x: str(x).replace("10000", ""))
merged_updated_df['begin_year'] = pd.to_numeric(merged_updated_df['begin_year'], errors='coerce')
merged_updated_df['begin_year'] = merged_updated_df['begin_year'].astype('Int64')
merged_updated_df['begin_year'] = merged_updated_df['begin_year'].replace(0, np.nan)

# merged_updated_df['begin_volume'] = None
# merged_updated_df['end_volume'] = None
merged_updated_df['ignore_warnings'] = "Yes"
merged_updated_df = merged_updated_df[((~merged_updated_df['begin_year'].isna()) | (~merged_updated_df['begin_volume'].isna()) | (merged_updated_df['record_type'] == "HOLDING"))]
# merged_updated_df = merged_updated_df[(merged_updated_df['nlm_unique_id'] != "NLM_0052457") & (merged_updated_df['nlm_unique_id'] != "NLM_0154720") & (merged_updated_df['nlm_unique_id'] != "NLM_0255562") & (merged_updated_df['nlm_unique_id'] != "NLM_0330471") & (merged_updated_df['nlm_unique_id'] != "NLM_0372435")  & (merged_updated_df['nlm_unique_id'] != "NLM_0326264")]
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

# print("\n\nafter remove bibliographic lifecycle\n\n")
#
# print(different_ranges_alma_output_df)
try:
    merged_updated_df = merged_updated_df.drop('index', axis=1)

except:
    "index column already removed"
try:
    merged_updated_df = merged_updated_df.drop('libid', axis=1)

except:
    "libid column already removed"


merged_updated_df.to_csv('Output/Update Final.csv', index=False)
