import streamlit as st
import pandas as pd
from apyori import apriori
import os
import pyarrow.parquet as pq

# Load example data
# Specify the folder path
folder_path = 'C:/Users/PRASAD/Desktop/Main_Project/Output/FPGrowth/association_rules'

# List all files in the folder
file_list = [x for x in os.listdir(folder_path) if x.endswith('.parquet')]

# Load Parquet files
data_frames = []
for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    table = pq.read_table(file_path)
    data_frame = table.to_pandas()
    data_frames.append(data_frame)

# Concatenate data frames if needed
df = pd.concat(data_frames, ignore_index=True)
df = df.sort_values("confidence")[::-1]

list1 = [x[0] for x in df["antecedent"]]
set1 = set(list1)
list1 = list(set1)
# Preprocess the data
# transactions = df.groupby('TransactionID')['Item'].apply(list).tolist()

# Sidebar
st.sidebar.header('Item Prediction')

selected_item = st.sidebar.selectbox('Select an item:', list1)

# Display selected item
st.sidebar.write('You selected:', selected_item)

# Apriori algorithm using apyori
association_rules = df

# Predict related items
related_items = []

for j in range(len(df["antecedent"])):
    if (df["antecedent"][j][0] == selected_item):
        related_items.append(df["consequent"][j][0])

# Remove duplicates and selected item from related items
related_items = list(set(related_items))

# Display predicted items
st.subheader('Predicted related items:')
for item in related_items:
    st.write(item)
