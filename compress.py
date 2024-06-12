import pandas as pd

def merge_intervals_optimized(df):
    # Sort the DataFrame
    df.sort_values(by=['nlm_unique_id', 'holdings_format', 'record_type', 'begin_year', 'end_year', 'embargo_period'], ascending=[True, True, True, True, True, True], inplace=True)
    # Using 10000 to represent 'indefinite'

    df.loc[df['record_type'] == 'RANGE', 'end_year'] = df.loc[df['record_type'] == 'RANGE', 'end_year'].fillna(10000)


    output_df = pd.DataFrame(columns=df.columns)
    current_row = None


    # df = df.fillna(np.nan)
    for _, row in df.iterrows():


        if row['record_type'] == 'HOLDING':
            print("holding")
            output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)
        elif row['record_type'] == 'RANGE':

            print("range")
            if current_row is not None:
                print("current exists")
                # if row['embargo_period']) == current_row['embargo_period'] and
                if row['nlm_unique_id'] != current_row['nlm_unique_id'] or \
                    (row['nlm_unique_id'] == current_row['nlm_unique_id'] and row['holdings_format'] != current_row['holdings_format']):
                    output_df = pd.concat([output_df, pd.DataFrame([current_row])], ignore_index=True)
                    current_row = row
                    print("first range in title.  dump current of last")
                if row['nlm_unique_id'] == current_row['nlm_unique_id'] and \
                    row['holdings_format'] == current_row['holdings_format'] and \
                    row['begin_year'] > current_row['end_year']:
                    left_effective_date=row['end_year'] - ((row['embargo_period'] / 12))
                    right_effective_date=current_row['end_year'] - ((current_row['embargo_period'] / 12))
                    if current_row['embargo_period'] != 0 and current_row['embargo_period'] is not None and right_effective_date > left_effective_date:
                        current_row['embargo_period'] = 0
                    output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)
                    print("additional range in title.   begin year of incoming greater than current")
                elif row['nlm_unique_id'] == current_row['nlm_unique_id'] and \
                    row['holdings_format'] == current_row['holdings_format'] and \
                    row['begin_year'] <= current_row['end_year'] and \
                    row['embargo_period'] != current_row['embargo_period']:
                    print("additional range in title.   begin year of incoming less than current.  and embargoes not equal")
                    left_effective_date=row['end_year'] - ((row['embargo_period'] / 12))
                    right_effective_date=current_row['end_year'] - ((current_row['embargo_period'] / 12))
                    if left_effective_date > right_effective_date:
                        print("additional range in title.   begin year of incoming less than current.  and embargoes not equal.   choose incoming end year")
                        current_row['end_year'] = row['end_year']
                        current_row['embargo_period'] = row['embargo_period']
                    elif left_effective_date == right_effective_date and current_row['embargo_period'] > row['embargo_period']:
                        print("additional range in title.   begin year of incoming less than current.  and embargoes not equal but effective end dates equal.   set current")
                        left_embargo_period = row['embargo_period']
                        right_embargo_period  = current_row['embargo_period']

                        if left_embargo_period < right_embargo_period:
                            current_row['end_year'] = row['end_year']
                            current_row['embargo_period'] = row['embargo_period']
                            print("set current embargo period to incoming if it's less.")
                elif row['nlm_unique_id'] == current_row['nlm_unique_id'] and \
                    row['holdings_format'] == current_row['holdings_format'] and \
                    row['begin_year'] <= current_row['end_year'] and \
                    row['embargo_period'] == current_row['embargo_period']:
                    print("begin year of incoming less than current.  embargo periods are the same")
                    # Extend the current range if overlapping
                    current_row['end_year'] = max(current_row['end_year'], row['end_year'])

                     # elif elif row['nlm_unique_id'] == current_row['nlm_unique_id'] and
                     #     row['holdings_format'] == current_row['holdings_format'] and
                     #     row['begin_year'] > current_row['end_year']:
                     #     #row['embargo_period']) == current_row['embargo_period']:
                     #
                     #     current_row['end_year'] = max(current_row['end_year'], row['end_year'])

            else:
                print("set current")
                current_row = row

                #output_df = pd.concat([output_df, pd.DataFrame([row])], ignore_index=True)
    # Append the last range row if it exists
    if current_row is not None and current_row['record_type'] == 'RANGE':
        output_df=pd.concat([output_df, pd.DataFrame([current_row])], ignore_index=True)

    return output_df
current_alma_df = pd.read_csv('Processing/compressed_before_optimization_test.csv', dtype={'begin_year': 'Int64', 'end_year': 'Int64', 'begin_volume': 'Int64', 'end_volume': 'Int64', 'nlm_unique_id': 'str'}, engine='python')

current_alma_compressed_df = merge_intervals_optimized(current_alma_df.copy())

current_alma_compressed_df.to_csv('Processing/compressed_after_optimization.csv', index=False)
