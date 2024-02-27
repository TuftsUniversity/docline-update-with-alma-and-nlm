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

    for index, row in df.iterrows():
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
