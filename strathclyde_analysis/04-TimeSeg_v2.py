import pickle as pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import copy as copy

plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams["figure.dpi"] = 120


# Load pickled data

# In[2]:


pickle_path = "pickles"


# In[3]:


sensor_data = pickle.load(open(pickle_path + "/" + "strath_inputs_v2.p", "rb"))
cmm_data = pickle.load(open(pickle_path + "/" + "strath_outputs_v2.p", "rb")).values

# split into forging, heating, transfer phases

stitched_data = sensor_data[0:]

stitched_data = np.concatenate(stitched_data, axis=0)

column_names = sensor_data[0].columns

# segment based on digital signals of Heat and Force
digital_heat = np.diff(stitched_data[:, -1])
digital_forge = np.diff((stitched_data[:, 3] > 0).astype("int"))

print(np.argwhere(column_names == "$U_GH_HEATON_1 (U25S0)"))
print(np.argwhere(column_names == "Force [kN]"))


digital_heat_diff_index = np.argwhere(digital_heat > 0)
digital_forge_start_index = np.argwhere(digital_forge == 1)
digital_forge_end_index = np.argwhere(digital_forge == -1)

# for
heating_traces = [
    stitched_data[digital_heat_diff_index[i][0] : digital_heat_diff_index[i + 1][0]]
    for i in range(digital_heat_diff_index.shape[0])
    if i < (digital_heat_diff_index.shape[0] - 1)
]
forging_traces = [
    stitched_data[digital_forge_start[0] : digital_forge_end[0]]
    for digital_forge_start, digital_forge_end in zip(
        digital_forge_start_index, digital_forge_end_index
    )
]


# verify the number of parts segmented
if len(heating_traces) != len(sensor_data):
    print("STITCHING ERROR IN HEATING PHASE")
if len(forging_traces) != len(sensor_data):
    print("STITCHING ERROR IN FORGING PHASE")

# =============PICKLE THEM========
pickle_path = "pickles"


if pickle_path not in os.listdir():
    os.mkdir(pickle_path)

# save into pickle file
pickle.dump(heating_traces, open(pickle_path + "/" + "heating_inputs.p", "wb"))
pickle.dump(forging_traces, open(pickle_path + "/" + "forging_inputs.p", "wb"))
pickle.dump(column_names, open(pickle_path + "/" + "column_names.p", "wb"))
