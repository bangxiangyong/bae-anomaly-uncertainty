
# ENSURE THAT PICKLES ARE NOT DUPLICATED/ERASED PER MODEL

import pandas as pd
import numpy as np

# file_path = "results/ZEMA_UNCOOD/ZEMA_HYD_UNCOOD_retained_perf.csv"
# file_path = "results/STRATH_UNCOOD/STRATH_FORGE_UNCOOD_retained_perf.csv"
file_path = "ZEMA_HYD_UNCOODV2_retained_perf (12).csv"

data = pd.read_csv(file_path)

pickles_unique = np.unique(data["pickle"])

pickle_lens = []
for pickle_uniq in pickles_unique:
    num_pickles = data[data["pickle"]==pickle_uniq]
    pickle_lens.append(len(num_pickles))

print(np.diff(pickle_lens).sum())