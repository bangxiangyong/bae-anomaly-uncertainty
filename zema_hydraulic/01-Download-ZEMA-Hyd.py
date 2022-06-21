import os, requests, zipfile, io
import numpy as np
import pickle

data_url = "https://zenodo.org/record/1323611/files/data.zip?download=1"


def download_and_extract(url, destination, force=False):
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
            print("Zip file was not extracted")
            return

    zipDocument.extractall(destination)


download_and_extract(data_url, "dataset/ZEMA_Hydraulic/")


data_path = "dataset/ZEMA_Hydraulic/"

filenames_input_data_1Hz = ["ts1", "ts2", "ts3", "ts4", "vs1", "se", "ce", "cp"]
filenames_input_data_1Hz = [file.upper() + ".txt" for file in filenames_input_data_1Hz]

filenames_input_data_10Hz = ["fs1", "fs2"]
filenames_input_data_10Hz = [
    file.upper() + ".txt" for file in filenames_input_data_10Hz
]

filenames_input_data_100Hz = ["ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "eps1"]
filenames_input_data_100Hz = [
    file.upper() + ".txt" for file in filenames_input_data_100Hz
]

data_input_data_1Hz = np.zeros((2205, 60, len(filenames_input_data_1Hz)))
data_input_data_10Hz = np.zeros((2205, 600, len(filenames_input_data_10Hz)))
data_input_data_100Hz = np.zeros((2205, 6000, len(filenames_input_data_100Hz)))

for id_, file_name in enumerate(filenames_input_data_1Hz):
    input_data = np.loadtxt(data_path + file_name, delimiter="\t")
    data_input_data_1Hz[:, :, id_] = input_data.copy()

for id_, file_name in enumerate(filenames_input_data_10Hz):
    input_data = np.loadtxt(data_path + file_name, delimiter="\t")
    data_input_data_10Hz[:, :, id_] = input_data.copy()

for id_, file_name in enumerate(filenames_input_data_100Hz):
    input_data = np.loadtxt(data_path + file_name, delimiter="\t")
    data_input_data_100Hz[:, :, id_] = input_data.copy()

# deal with output data now
filename_target_data = "profile.txt"
data_path = "dataset/ZEMA_Hydraulic/"


targets_data = np.loadtxt(data_path + filename_target_data, delimiter="\t").astype(int)[
    :, :-1
]
target_dim_maps = {
    "cooler": [100, 20, 3],
    "valve": [100, 90, 80, 73],
    "pump": [0, 1, 2],
    "acc": [130, 115, 100, 90],
}

coded_targets_data = np.copy(targets_data)

for dim_i in range(targets_data.shape[1]):
    maps = list(target_dim_maps.values())[dim_i]
    for pos_i, map_i in enumerate(maps):
        coded_targets_data[
            np.argwhere(targets_data[:, dim_i] == map_i)[:, 0], dim_i
        ] = pos_i

all_tensor_output = np.copy(coded_targets_data)


# save raw data into dict
raw_data = {
    "Hz_1": data_input_data_1Hz,
    "Hz_10": data_input_data_10Hz,
    "Hz_100": data_input_data_100Hz,
    "target": all_tensor_output,
}

# Move Axis
for id_, key in enumerate(["Hz_1", "Hz_10", "Hz_100"]):
    raw_data[key] = np.moveaxis(raw_data[key], 1, 2)

pickle_folder = "pickles"

if os.path.exists(pickle_folder) == False:
    os.mkdir(pickle_folder)

# Pickle them
pickle.dump(raw_data, open(pickle_folder + "/raw_data.p", "wb"))
