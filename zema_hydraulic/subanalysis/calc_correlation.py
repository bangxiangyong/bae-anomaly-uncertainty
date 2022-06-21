import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from pandas.plotting import scatter_matrix
from bnn.seed import bae_set_seed

bae_set_seed(100)

plt.style.use('ggplot')

total_sensors = 17
sensor_names = ["ts1","ts2","ts3","ts4","vs1","se","ce","cp"]+["fs1","fs2"]+["ps1","ps2","ps3","ps4","ps5","ps6","eps1"]

pickle_path="pickles/"
data_raw = pickle.load(open(pickle_path+"data_ft_resampled.p", "rb" ) )
x_train, x_test, x_ood, x_test_noise,x_test_drift, = data_raw['x_train'], data_raw['x_test'], data_raw['x_ood'], data_raw['x_test_noise'], data_raw['x_test_drift']
y_train, y_test, y_ood = data_raw['y_train'], data_raw['y_test'], data_raw['y_ood']

def get_corr_data(x_data,label="train"):
    #put into df
    x_flatten = x_ood.reshape(-1,17)
    df = pd.DataFrame(x_flatten)
    df.columns = sensor_names
    #correlation
    correlation_table = df.corr()
    print(correlation_table)
    plt.matshow(correlation_table)
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df.columns)), df.columns)
    plt.colorbar()
    plt.show()
    plt.savefig(label+"_correlation_table.png")

    #save csv
    correlation_table.to_csv(label+"_correlation_table.csv")

    return correlation_table

get_corr_data(x_train,"healthy")
get_corr_data(x_ood,"faulty")

for num_sensor in range(len(x_test_noise)):
    for id_,noise_level in enumerate(('0', '5', '10', '15', '20', '25')):
        if num_sensor == 10:
            get_corr_data(x_test_noise[num_sensor][id_],"noise"+str(noise_level)+"_sensor"+str(num_sensor))

for num_sensor in range(len(x_test_drift)):
    for id_,drift_level in enumerate(('0', '5', '10', '15', '20', '25')):
        if num_sensor == 10:
            get_corr_data(x_test_drift[num_sensor][id_],"drift"+str(drift_level)+"_sensor"+str(num_sensor))


#put into df
x_flatten = x_ood.reshape(-1,17)
df = pd.DataFrame(x_flatten)
df.columns = sensor_names

#plot correlation
sensor_i1 =3
sensor_i2 =1
num_sample=0
plt.figure()
plt.scatter(x_flatten[:,sensor_i1].flatten(),x_flatten[:,sensor_i2].flatten())


#entire scatter matrix
#very computational intensive
plot_full_scatter = False
if plot_full_scatter:
    plt.figure()
    scatter_matrix(df, figsize=(8, 8), diagonal='kde')
    plt.show()

#correlation
correlation_table = df.corr()
print(correlation_table)
plt.matshow(correlation_table)
plt.xticks(range(len(df.columns)), df.columns)
plt.yticks(range(len(df.columns)), df.columns)
plt.colorbar()
plt.show()
plt.savefig("correlation_table.png")

#save csv
correlation_table.to_csv("correlation_table.csv")

