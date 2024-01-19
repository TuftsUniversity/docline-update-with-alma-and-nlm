import pandas as pd
def merge_intervals_optimized(df):
    df.sort_values(by=['nlm_unique_id', 'holdings_format', 'action', 'begin_year'], inplace=True)
    df['end_year'].fillna(10000, inplace=True)  # Using 10000 to represent 'indefinite'

    merged_intervals = []
    current_start = current_end = None
    current_id = current_format = current_action = None

    for index, row in df.iterrows():
        if row['record_type'] == "HOLDING":
            merged_intervals.append([row['nlm_unique_id'], row['holdings_format'], row['action'], row['record_type'], row['begin_year'], row['end_year']])
        else:
            if pd.isna(row['begin_year']):
                # Skip rows where begin_year is NaN
                continue

            # Check if we have moved to a new group (nlm_unique_id, holdings_format, action)
            if row['nlm_unique_id'] != current_id or row['holdings_format'] != current_format or row['action'] != current_action:
                # Append the previous group's merged interval if it exists
                if current_id is not None:
                    merged_intervals.append([current_id, current_format, current_action, row['record_type'], current_start, None if current_end == 10000 else current_end])
                # Start a new group
                current_id, current_format, current_action, current_start, current_end = row['nlm_unique_id'], row['holdings_format'], row['action'], row['begin_year'], row['end_year']
            else:
                # Merge intervals within the same group
                if current_end != 10000 and (row['begin_year'] <= current_end or row['begin_year'] - 1 == current_end):
                    current_end = max(current_end, row['end_year'])
                else:
                    # Append the previous interval and start a new one within the same group
                    merged_intervals.append([current_id, current_format, current_action, row['record_type'], current_start, None if current_end == 10000 else current_end])
                    current_start, current_end = row['begin_year'], row['end_year']

    # Append the last interval
    #merged_intervals.append([current_id, current_format, current_action, row['record_type'], current_start, None if current_end == 10000 else current_end])

    merged_df = pd.DataFrame(merged_intervals, columns=['nlm_unique_id', 'holdings_format', 'action', 'record_type' ,'begin_year', 'end_year'])
    return merged_df
alma_only_add_df = pd.read_csv("Output/Test Input 1.csv", dtype={'begin_year': "Int64", 'end_year': "Int64"}, engine='python')
#
# merged_df = merge_intervals_optimized(df)
# merged_df.head(10)  # Displaying the first 10 rows of the merged DataFrame

#
# alma_only_add_df = alma_only_add_df.sort_values(by = ['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year'], ascending = [True, True, False, True, True, True], na_position = 'first')
filtered_new_df = alma_only_add_df[['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year', 'end_year']]#.dropna(subset=['begin_year'])
merged_new_df_optimized = merge_intervals_optimized(filtered_new_df.copy())

print(merged_new_df_optimized)
# Merge the original dataframe with the merged intervals
updated_df = alma_only_add_df.drop(columns=['begin_year', 'end_year']).merge(merged_new_df_optimized, on=['nlm_unique_id', 'holdings_format', 'action', 'record_type'], how='left')


# Optional: Remove duplicate rows based on the compound key
updated_df.drop_duplicates(subset=['nlm_unique_id', 'holdings_format', 'action', 'record_type'], inplace=True)

updated_df.to_csv("Output/Test3.csv", index=False)
