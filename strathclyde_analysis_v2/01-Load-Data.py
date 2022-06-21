#!/usr/bin/env python
# coding: utf-8

# # Load Data
#

# We download the AFRC radial forge data from the link specified.
#
# Credits to Christos Tachtatzis for the code to download & extract.

# In[2]:

afrc_data_url = "https://zenodo.org/record/3405265/files/STRATH%20radial%20forge%20dataset%20v2.zip?download=1"

data_path = "Data_v2"  # folder for dataset

# In[3]:

import os
import io
import requests
import zipfile


def download_and_extract(url, destination, force=False):
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    response = requests.get(url)
    zipDocument = zipfile.ZipFile(io.BytesIO(response.content))

    # Attempt to see if we are going to overwrite anything
    if not force:
        abort = False
        for file in zipDocument.filelist:
            if os.path.isfile(os.path.join(destination, file.filename)):
                print(
                    file.filename,
                    "already exists. If you want to overwrite the file call the method with force=True",
                )
                abort = True
        if abort:
            print("Zip file was not extracted.")
            return

    zipDocument.extractall(destination)


# In[4]:

download_and_extract(afrc_data_url, data_path)


# The data is downloaded into the folder 'Data' , now we transform the data into a list of dataframes.
#
# Each dataframe in list represents the time-series measurements of all sensors for a part.

# In[2]:


import pandas as pd
import pickle as pickle
import os
import numpy as np

# ## Load sensor data into dataframes

# In[6]:


data_inputs_list = []

# load each part's data as a dataframe to a list
for filename in os.listdir(
    os.path.join(data_path, "STRATH radial forge dataset 11Sep19")
):
    if "Scope" in filename and "csv" in filename:
        file_csv = pd.read_csv(
            os.path.join(data_path, "STRATH radial forge dataset 11Sep19", filename),
            encoding="cp1252",
        )
        data_inputs_list.append(file_csv)


# In[8]:


len(data_inputs_list)


# ## Load CMM data into dataframe
#
# 1. Read data
# 2. Subtract the CMM measurements from the "base value"
# 3. Save into a dataframe

# In[4]:


data_path = "Data_v2"  # folder for dataset
output_pd = pd.read_excel(
    os.path.join(data_path, "STRATH radial forge dataset 11Sep19", "CMMData.xlsx")
)

# extract necessary output values
output_headers = output_pd.columns[4:]
base_val = output_pd.values[0, 4:]

output_val = output_pd.values[3:, 4:]

np_data_outputs = output_val

# extract abs error from expected base values
for output in range(np_data_outputs.shape[1]):
    np_data_outputs[:, output] -= base_val[output]
np_data_outputs = np.abs(np_data_outputs)


# In[5]:


output_df = {}
for i, value in enumerate(output_headers):
    new_df = {value: np_data_outputs[:, i]}
    output_df.update(new_df)
output_df = pd.DataFrame(output_df)


# In[6]:

output_df


# ## Pickle Data
#
# Pickle the input & output data for ease of future use

# In[13]:

pickle_path = "pickles"
input_file_name = "strath_inputs_v2.p"
output_file_name = "strath_outputs_v2.p"


if pickle_path not in os.listdir():
    os.mkdir(pickle_path)

# save into pickle file
pickle.dump(data_inputs_list, open(pickle_path + "/" + input_file_name, "wb"))
pickle.dump(output_df, open(pickle_path + "/" + output_file_name, "wb"))

print(
    "Data preparation from Zenodo completed as "
    + input_file_name
    + " and "
    + output_file_name
)
