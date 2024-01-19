import pandas as pd
import numpy as np
import time
import os
import sys

docline_df = pd.read_csv("Output/Update Final.csv", engine='python')
#docline_df[docline_df['end_year'].isna() | docline_df['end_year'] == "" ]['currently_received'] = "Yes"
pd.set_option('display.max_columns', None)
print(docline_df.loc[0:2,'end_year'])

print(docline_df[docline_df['end_year'].isna()])


docline_df.to_excel("Output/Initial Test.xlsx", index=False)
new_docline_df = docline_df.copy()


# print(docline_df.loc[2, 'currently_received'].type)
# print(docline_df.loc[2, 'end_year'].type)
#pr#int(docline_df[docline_df['nlm_unique_id'] == 'NLM_0004041']['end_year'].type)

#print(docline_df[docline_df['nlm_unique_id'] == 'NLM_0004041']['currently_received'])
# docline_df[docline_df['end_year'].isnull()]['currently_received'] = "Yes"

new_docline_df.loc[new_docline_df['end_year'].isna(), 'currently_received'] = "Yes"

for index, row in new_docline_df.iterrows():
    if row['record_type'] == "RANGE":
        new_docline_df.loc[index - 1, 'currently_received'] = row['currently_received']

# Display the DataFrame to verify the changes



new_docline_df.to_excel("Output/Updated.xlsx", index=False)
print(docline_df)
