import pandas as pd
import sys
def merge_intervals_optimized(df):
    # Sort the DataFrame
    df.sort_values(by=['nlm_unique_id', 'holdings_format', 'action', 'begin_year'], inplace=True)
    df['end_year'].fillna(10000, inplace=True)  # Using 10000 to represent 'indefinite'

    output_df = pd.DataFrame(columns=df.columns)
    current_row = None

    for _, row in df.iterrows():
        if row['record_type'] == 'HOLDING':
            output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)
        elif row['record_type'] == 'RANGE':
            if current_row is None or \
               row['nlm_unique_id'] != current_row['nlm_unique_id'] or \
               row['holdings_format'] != current_row['holdings_format'] or \
               row['begin_year'] > current_row['end_year']:
                if current_row is not None:
                    output_df = pd.concat([output_df, pd.DataFrame([current_row])], ignore_index=True)
                current_row = row
            else:
                # Extend the current range if overlapping
                current_row['end_year'] = max(current_row['end_year'], row['end_year'])

    # Append the last range row if it exists
    if current_row is not None and current_row['record_type'] == 'RANGE':
        output_df = pd.concat([output_df, pd.DataFrame([current_row])], ignore_index=True)

    return output_df

# Usage
alma_only_add_df = pd.read_csv("Output/Test Input 1.csv", dtype={'begin_year': "Int64", 'end_year': "Int64"}, engine='python')
updated_df = merge_intervals_optimized(alma_only_add_df)
updated_df.loc[updated_df['end_year'] == 10000, 'currently_received'] = "Yes"

# print(updated_df.loc[updated_df['end_year'] == ''])
# print(updated_df['end_year'])
# sys.exit()
for index, row in updated_df.iterrows():
    if index != 0:
        if row['record_type'] == "RANGE" and updated_df.loc[index - 1, 'record_type'] == "HOLDING":
            updated_df.loc[index - 1, 'currently_received'] = row['currently_received']

updated_df.sort_values(by=['nlm_unique_id', 'holdings_format', 'action', 'record_type', 'begin_year'], inplace=True)
updated_df.to_csv("Output/Test3.csv", index=False)
