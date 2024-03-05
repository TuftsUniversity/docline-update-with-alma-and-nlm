import pandas as pd

# Load the CSV files
update_final_df = pd.read_csv('Output/Update Final.csv')
removed_lines_compare_df = pd.read_csv('Output/removed_lines-compare.csv')

# Create a boolean mask for rows in 'update_final_df' that match the 'nlm_unique_id' and 'holdings_format'
# in 'removed_lines_compare_df'
mask = update_final_df.set_index(['nlm_unique_id', 'holdings_format']).index.isin(removed_lines_compare_df.set_index(['nlm_unique_id', 'holdings_format']).index)

# Apply the mask to 'update_final_df' to get only matched rows
matched_rows_update_final_df = update_final_df[mask]

# Save the matched rows to a new CSV file
matched_rows_output_file_path = 'Output/matched_rows_from_Update_Final.csv'
matched_rows_update_final_df.to_csv(matched_rows_output_file_path, index=False)

print(f"Matched rows saved to {matched_rows_output_file_path}")
