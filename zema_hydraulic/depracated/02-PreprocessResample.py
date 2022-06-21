import pickle
from scipy.signal import resample
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy as copy
from bnn.seed import bae_set_seed

bae_set_seed(100)

#args for script
column_target = 0 #0-4
index_healthy = 0
total_sensors = 17
path_data = "pickles/raw_data.p"
pickle_path="pickles/"
data_raw = pickle.load( open( pickle_path+"raw_data.p", "rb" ) )
num_cycles = 2205
n_resample = 60
data_raw_resampled = {"Hz_1":np.ones((num_cycles,n_resample,data_raw['Hz_1'].shape[-1])),
                      "Hz_10":np.ones((num_cycles,n_resample,data_raw['Hz_10'].shape[-1])),
                      "Hz_100":np.ones((num_cycles,n_resample,data_raw['Hz_100'].shape[-1])),"target":data_raw['target']}

#resample data to n Hz
for id_, key in enumerate(["Hz_1","Hz_10","Hz_100"]):
    for sensor_id in range(data_raw[key].shape[-1]):
        data_raw_resampled[key][:,:,sensor_id] = resample(data_raw[key][:,:,sensor_id],n_resample,axis=1)

#apply filter here to get only healthy data for train,test, while ood will be everything else
index_iod = np.argwhere(data_raw_resampled['target'][:,column_target]==index_healthy).reshape(-1) #iod index
index_ood = np.argwhere(data_raw_resampled['target'][:,column_target]!=index_healthy).reshape(-1) #ood index

for id_, key in enumerate(["Hz_1","Hz_10","Hz_100","target"]):
    if id_ == 0:
        data_iod = {key:data_raw_resampled[key][index_iod]}
        data_ood = {key:data_raw_resampled[key][index_ood]}
    else:
        data_iod.update({key:data_raw_resampled[key][index_iod]})
        data_ood.update({key:data_raw_resampled[key][index_ood]})


# split data
train_index, test_index = train_test_split(np.arange(len(index_iod)), test_size=0.3,random_state=0, shuffle=True)

for id_, key in enumerate(["Hz_1","Hz_10","Hz_100"]):
    if id_ == 0:
        x_train_raw = {key:data_iod[key][train_index]}
        x_test_raw = {key:data_iod[key][test_index]}
        x_ood_raw = {key:data_ood[key]}
    else:
        x_train_raw.update({key:data_iod[key][train_index]})
        x_test_raw.update({key:data_iod[key][test_index]})
        x_ood_raw.update({key:data_ood[key]})

#choose one sensor to create noisy/drifty dataset
#create noisy x_test data
noise_levels = [0,0.05,0.10,0.15,0.2,0.25]

def convert_sensor_index(selected_index=0):
    """
    Given 0-17, converts to key and index for accessing array within dictionary
    e.g "selected_index = 10 yields: index=1, key ='Hz_10'
    """
    sensor_index = 0
    total_sensors = 0
    total_sensors_index = []
    key_list = ["Hz_1","Hz_10","Hz_100"]
    for key_id, key in enumerate(key_list):
        total_sensors += data_raw[key].shape[-1]

        if selected_index < total_sensors:
            previous_index = (total_sensors-data_raw[key].shape[-1])
            return selected_index-previous_index, key

    return 0,0

def add_noise_single(x_raw,noise_level,index_noisy_sensor):
    x_raw_temp = copy.deepcopy(x_raw)
    index_noisy_sensor_temp, key_ = convert_sensor_index(index_noisy_sensor)
    x_signal = x_raw_temp[key_][:,:,index_noisy_sensor_temp]
    noise_volts = np.random.uniform(1-noise_level, 1+noise_level, size=x_signal.shape)
    x_signal_noisy = x_signal * noise_volts
    x_raw_temp[key_][:,:,index_noisy_sensor_temp] = x_signal_noisy
    return x_raw_temp

x_test_noise_raw = []
for sensor_index in range(total_sensors):
    x_test_noise_raw.append([add_noise_single(x_test_raw, noise_level, sensor_index) for noise_level in noise_levels])

#create drifting x_test data
drift_levels = [0,0.05,0.10,0.15,0.2,0.25]

def add_drift_single(x_raw,drift_level,index_noisy_sensor,mean=False):
    x_raw_temp = copy.deepcopy(x_raw)
    index_noisy_sensor_temp, key_ = convert_sensor_index(index_noisy_sensor)
    x_signal = x_raw_temp[key_][:,:,index_noisy_sensor_temp]
    if mean:
        x_signal_noisy = x_signal + np.dstack([x_signal.mean(1)]*x_signal.shape[-1])*drift_level

    else:
        x_signal_noisy = x_signal + drift_level
    x_raw_temp[key_][:,:,index_noisy_sensor_temp] = x_signal_noisy
    return x_raw_temp

x_test_drift_raw = []
for sensor_index in range(total_sensors):
    x_test_drift_raw.append([add_drift_single(x_test_raw, drift_level, sensor_index,mean=True) for drift_level in drift_levels])

y_train,y_test,y_ood = data_iod["target"][train_index],data_iod["target"][test_index],data_ood["target"]

#collapse dict into array
def collapse_keys(x_raw):
    """
    Collapses dict of keys ["Hz_1","Hz_10","Hz_100"] into a single numpy array
    """
    x_raw_ft = np.append(x_raw['Hz_1'],values=x_raw['Hz_10'],axis=2)
    x_raw_ft = np.append(x_raw_ft,values=x_raw['Hz_100'],axis=2)
    return x_raw_ft

x_train_ft = collapse_keys(x_train_raw)
x_test_ft = collapse_keys(x_test_raw)
x_ood_ft = collapse_keys(x_ood_raw)
x_test_noise_ft = []
x_test_drift_ft = []

for sensor_index in range(total_sensors):
    print(sensor_index)
    x_test_noise_ft.append([collapse_keys(noise_data) for noise_data in x_test_noise_raw[sensor_index]])
    x_test_drift_ft.append([collapse_keys(drift_data) for drift_data in x_test_drift_raw[sensor_index]])

data_ft = {"x_train":x_train_ft,
           "x_test":x_test_ft,
           "x_ood":x_ood_ft,
           "x_test_noise":x_test_noise_ft,
           "x_test_drift":x_test_drift_ft,
           "y_train":y_train,
           "y_test":y_test,
           "y_ood":y_ood
           }

for key in data_ft.keys():
    try:
        print(key+str(data_ft[key].shape))
    except Exception as e :
        print(key+str(len(data_ft[key])))

pickle_folder= "pickles"
pickle.dump(data_ft, open(pickle_folder+"/data_ft_resampled.p", "wb"))

# Plot samples
plot_sample_index = 10
plot_sensor_index = 10
plt.figure()
plt.plot(x_train_ft[plot_sample_index,:,plot_sensor_index])
plt.plot(x_test_noise_ft[plot_sensor_index][-1][plot_sample_index,:,plot_sensor_index])
plt.plot(x_test_drift_ft[plot_sensor_index][-1][plot_sample_index,:,plot_sensor_index])
plt.legend(["NORMAL","NOISE","DRIFT"])


#
pickle_folder= "pickles"
pickle.dump(data_ft, open(pickle_folder+"/data_ft.p", "wb"))

