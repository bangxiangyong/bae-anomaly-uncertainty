from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from bnn.ensembles_bnn import BAE_Ensemble_Manager
import pickle
import matplotlib.pyplot as plt
import numpy as np
from bnn.multiscaler import MultiScaler
import copy
from bnn.seed import bae_set_seed
from mpl_toolkits.mplot3d import Axes3D
import torch

bae_set_seed(100)

#LOAD DATA
total_sensors = 17
sensor_names = ["ts1","ts2","ts3","ts4","vs1","se","ce","cp"]+["fs1","fs2"]+["ps1","ps2","ps3","ps4","ps5","ps6","eps1"]
pickle_path="pickles/"
data_raw = pickle.load(open(pickle_path+"data_ft_resampled.p", "rb" ) )
x_train, x_test, x_ood, x_test_noise,x_test_drift, = data_raw['x_train'], data_raw['x_test'], data_raw['x_ood'], data_raw['x_test_noise'], data_raw['x_test_drift']
y_train, y_test, y_ood = data_raw['y_train'], data_raw['y_test'], data_raw['y_ood']

def flatten_dimensions(np_array):
    return np_array.reshape(np_array.shape[0],-1)

def nested_list_apply(x_list, apply_func, *args, **kwargs):
    x_list_result = []
    for sensor_data in x_list:
        temp_data = []
        for noise_data in sensor_data:
            temp_data.append(apply_func(noise_data, *args, **kwargs))
        x_list_result.append(temp_data)
    return x_list_result

#normalise data
scaler_class = MultiScaler
scaler = scaler_class(scaler_class=StandardScaler)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_ood = scaler.transform(x_ood)

x_test_noise = nested_list_apply(x_test_noise, scaler.transform)
x_test_drift = nested_list_apply(x_test_drift, scaler.transform)

#reshape to flatten last two dimensions to fit bnn
x_train = x_train.reshape(x_train.shape[0],-1)
x_test = x_test.reshape(x_test.shape[0],-1)
x_ood = x_ood.reshape(x_ood.shape[0],-1)
x_test_noise = nested_list_apply(x_test_noise, flatten_dimensions)
x_test_drift = nested_list_apply(x_test_drift, flatten_dimensions)

#define model
optimiser = "Adam"
num_epoch_mu = 100 #10000 #1000
architecture_mu = [500,250,3,250,500]
architecture_sig_diag_cov=architecture_mu
lr = 0.0008
num_samples = 10
use_cuda = torch.cuda.is_available()

#diag_cov & combined mode
model_name = "diag_cov"
bae_model = BAE_Ensemble_Manager(architecture=architecture_mu, num_samples=num_samples,
                                 num_epoch=num_epoch_mu, learning_rate=lr, bottleneck_layer=2,
                                 task="regression", optimiser=optimiser, mode="diag_cov", use_cuda=use_cuda)

#fit model
bae_model.fit(x_train)

#predictions
result_test = bae_model.predict(x_test, mode=model_name)
result_ood = bae_model.predict(x_ood, mode=model_name)
result_noise = nested_list_apply(x_test_noise, bae_model.predict, mode=model_name)
result_drift = nested_list_apply(x_test_drift, bae_model.predict, mode=model_name)
result_test_mu = bae_model.predict(x_test, mode="mu")
result_ood_mu = bae_model.predict(x_ood, mode="mu")
result_noise_mu = nested_list_apply(x_test_noise, bae_model.predict, mode="mu")
result_drift_mu = nested_list_apply(x_test_drift, bae_model.predict, mode="mu")

#compute more accurate epistemic unc.
def unc_scale(result,seq_len=60,num_sens=17):
    raw_y_pred=result['raw'][0]
    scaled_raw_y_pred = []
    raw_y_cov=result['raw'][1]
    scaled_raw_y_cov = []
    for i in range(len(raw_y_pred)):
        scaled_raw_y_pred.append(scaler.inverse_transform(raw_y_pred[i,:,:].reshape(-1,seq_len,num_sens)))
        scaled_raw_y_cov.append(scaler.inverse_variance((raw_y_cov[i,:,:]**0.5).reshape(-1,seq_len,num_sens)))
    scaled_raw_y_pred= np.array(scaled_raw_y_pred)
    scaled_raw_y_cov= np.array(scaled_raw_y_cov)
    epi_result = scaled_raw_y_pred.std(0)
    alea_result = scaled_raw_y_cov.mean(0)
    total_unc = epi_result+alea_result
    return epi_result, alea_result,total_unc

def alea_unc_scale(result,seq_len=60,num_sens=17):
    temp_res = copy.copy(result['y_cov'][0])
    temp_res= temp_res.reshape(-1,seq_len,num_sens)**0.5
    temp_res = scaler.inverse_variance(temp_res)
    return temp_res

def str_deci(number,num_deci=2):
    return ("{0:."+str(num_deci)+"f}").format(round(number,num_deci))

#plot reconstructed signal
plot_sample_index = 100
plot_sensor_index = 10
severity_index = 2
n_resample = 60

def get_recon_signal(mode="noise",plot_sample_index = 1,plot_sensor_index =12,n_resample = 60,severity_index = 0):
    # feature_sensor_indices = range((n_resample*plot_sensor_index),(n_resample*(1+plot_sensor_index)))
    if mode == "noise":
        recon_sig=scaler.inverse_transform(result_noise[plot_sensor_index][severity_index]['y_pred'][0].reshape(-1,n_resample,total_sensors))
        recon_sig_epi_unc,recon_sig_alea_unc,recon_sig_total_unc=unc_scale(result_noise[plot_sensor_index][severity_index])
        ori_sig = scaler.inverse_transform(x_test_noise[plot_sensor_index][severity_index].reshape(-1,n_resample,total_sensors))
    elif mode=="drift":
        recon_sig=scaler.inverse_transform(result_drift[plot_sensor_index][severity_index]['y_pred'][0].reshape(-1,n_resample,total_sensors))
        recon_sig_epi_unc,recon_sig_alea_unc,recon_sig_total_unc=unc_scale(result_drift[plot_sensor_index][severity_index])
        ori_sig = scaler.inverse_transform(x_test_drift[plot_sensor_index][severity_index].reshape(-1,n_resample,total_sensors))
    else:
        recon_sig=scaler.inverse_transform(result_ood['y_pred'][0].reshape(-1,n_resample,total_sensors))
        recon_sig_epi_unc,recon_sig_alea_unc,recon_sig_total_unc=unc_scale(result_ood)
        ori_sig = scaler.inverse_transform(x_ood.reshape(-1,n_resample,total_sensors))

    recon_loss = ((recon_sig - ori_sig)**2)

    plot_recon_sig = recon_sig[plot_sample_index,:,plot_sensor_index]
    plot_ori_sig = ori_sig[plot_sample_index,:,plot_sensor_index]
    plot_recon_loss = recon_loss[plot_sample_index,:,plot_sensor_index]
    epi_unc =recon_sig_epi_unc[plot_sample_index,:,plot_sensor_index]
    alea_unc =recon_sig_alea_unc[plot_sample_index,:,plot_sensor_index]
    total_unc = recon_sig_total_unc[plot_sample_index,:,plot_sensor_index]

    return {"recon_sig":plot_recon_sig,
            "ori_sig":plot_ori_sig,
            "recon_loss":plot_recon_loss,
           "epi_unc":epi_unc,
           "alea_unc":alea_unc,
           "total_unc":total_unc
            }

#PLOT RECONSTRUCTED SIGNAL
recon_signal_test = get_recon_signal(plot_sample_index = plot_sample_index,plot_sensor_index =plot_sensor_index,severity_index=0)
recon_signal_noise = get_recon_signal(mode="noise",plot_sample_index = plot_sample_index,plot_sensor_index =plot_sensor_index,severity_index=severity_index)
recon_signal_ood = get_recon_signal(mode="ood",plot_sample_index = plot_sample_index,plot_sensor_index =plot_sensor_index)
recon_signal_drift = get_recon_signal(mode="drift",plot_sample_index = plot_sample_index,plot_sensor_index =plot_sensor_index,severity_index=severity_index)


fig, axes = plt.subplots(2,2, figsize=(5,5),dpi=250)
axes = axes.reshape(-1)
plot_titles = ["a) Normal", "b) Cooler Condition (3%)", "c) Injected Noise (10%)","d) Injected Drift (10%)"]
ax_ids = np.arange(len(axes))
for ax_id,ax,recon_signal,plot_title in zip(ax_ids,axes,[recon_signal_test,recon_signal_ood,recon_signal_noise,recon_signal_drift],plot_titles):
    plot_recon_sig,plot_ori_sig,recon_loss,epi_unc,alea_unc,total_unc = recon_signal['recon_sig'],recon_signal['ori_sig'],recon_signal['recon_loss'],recon_signal['epi_unc'],recon_signal['alea_unc'],recon_signal['total_unc']
    ax.plot(plot_recon_sig)
    ax.plot(plot_ori_sig)
    ax.fill_between(range(n_resample), (plot_recon_sig+alea_unc), (plot_recon_sig-alea_unc),alpha=0.5,color='g')
    ax.fill_between(range(n_resample), (plot_recon_sig+epi_unc+alea_unc), (plot_recon_sig+alea_unc),alpha=0.5,color='r')
    ax.fill_between(range(n_resample), (plot_recon_sig-alea_unc), (plot_recon_sig-epi_unc-alea_unc),alpha=0.5,color='r')

    recon_loss_mean,epi_unc_mean, alea_unc_mean, total_unc_mean = recon_loss.mean(),epi_unc.mean(),alea_unc.mean(),epi_unc.mean()+alea_unc.mean()

    ax.legend(["Reconstructed","Measured","Aleatoric","Epistemic"],prop={'size': 6})
    if ax_id == 2 or ax_id == 3:
        ax.set_xlabel("Time(s)")
    if ax_id == 0 or ax_id == 2:
        ax.set_ylabel("Pressure (Bar)")
    ax.set_title(plot_title,fontsize=8)
    ax.text(0.35, 0.95,'Loss:'+str_deci(recon_loss_mean,2), ha='center', va='center', transform=ax.transAxes,fontsize=6)

#Reconstruction loss
def vanilla_recon_loss(x_test,result_mu):
    return ((result_mu['y_pred'][0]-x_test)**2).mean(1)

def get_nested_recon_loss(result,x_test):
    return [vanilla_recon_loss(x_test[id_],result[id_]) for id_, noise_data in enumerate(result)]

recon_test = vanilla_recon_loss(x_test,result_test_mu)
recon_ood = vanilla_recon_loss(x_ood,result_ood_mu)
recon_noise = []
recon_drift = []

for sensor_id,sensor_data in enumerate(result_noise_mu):
    temp_data_noise = []
    temp_data_drift = []
    for noise_id,noise_data in enumerate(sensor_data):
        temp_data_noise.append(vanilla_recon_loss(x_test_noise[sensor_id][noise_id],sensor_data[noise_id]))
        temp_data_drift.append(vanilla_recon_loss(x_test_drift[sensor_id][noise_id],sensor_data[noise_id]))
    recon_noise.append(temp_data_noise)
    recon_drift.append(temp_data_drift)

def mean_unc(result_data,seq_len=60,num_sens=17):
    epi,alea,total = unc_scale(result_data,seq_len=60,num_sens=17)
    epi = epi.reshape(-1,seq_len*num_sens)
    alea = alea.reshape(-1,seq_len*num_sens)
    total = total.reshape(-1,seq_len*num_sens)

    return epi.mean(1),alea.mean(1),total.mean(1)

epi_test,alea_test,_ = mean_unc(result_test)
epi_ood,alea_ood,_ = mean_unc(result_ood)
unc_result_noise = nested_list_apply(result_noise,mean_unc)
unc_result_drift = nested_list_apply(result_drift,mean_unc)

#aleatoric uncertainty
def mean_alea_unc(batch_cov_mat,index_alea=0):
    """
    For batch covariance matrix, extracts the diagonal and compute
    the mean by the last dimension (expected as the number of features)
    """
    batch_cov_mat_y_cov = batch_cov_mat['y_cov'][index_alea]
    if len(batch_cov_mat_y_cov.shape) ==3:
        #extract diagonal of cov. matrix
        iii,jjj = np.diag_indices(batch_cov_mat_y_cov.shape[-1])
        alea_temp = batch_cov_mat_y_cov[...,iii,jjj]
    else:
        alea_temp = batch_cov_mat_y_cov
    alea_temp = (alea_temp**0.5).mean(-1)
    return alea_temp

#argwhere to separate by severity of condition
index_target = 0
columns_ood = np.unique(y_ood[:,index_target])
index_ood = []

for unique_condition in columns_ood:
    index_ood += [np.argwhere(y_ood[:,index_target] == unique_condition).reshape(-1)]

#PLOTS FOR VARYING CONDITION
# plot - recon loss
show_outliers = False
recon_loss_plot =[]
recon_loss_plot += [recon_test]
for i in index_ood:
    recon_loss_plot += [recon_ood[i]]

# plot - epistemic uncertainty
epi_unc_plot =[]
epi_unc_plot += [epi_test]
for i in index_ood:
    epi_unc_plot += [epi_ood[i]]

#plot - aleatoric uncertainty (diagonal cov)
alea_unc_plot =[]
alea_unc_plot += [alea_test]
for i in index_ood:
    alea_unc_plot += [alea_ood[i]]

figsize = (6,3)
dpi = 250
fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=figsize,dpi=dpi)

x_tick_labels_eff = ('100', '20', '3')
ax1.boxplot(recon_loss_plot,showfliers=show_outliers)
ax1.set_xticklabels( x_tick_labels_eff)
ax1.set_ylabel("Reconstruction loss")
ax1.set_xlabel("Condition (%)")

ax2.boxplot(epi_unc_plot,showfliers=show_outliers)
ax2.set_xticklabels(x_tick_labels_eff)
ax2.set_ylabel("Epistemic uncertainty")
ax2.set_xlabel("Condition (%)")

ax3.boxplot(alea_unc_plot,showfliers=show_outliers)
ax3.set_xticklabels(x_tick_labels_eff)
ax3.set_ylabel("Aleatoric uncertainty")
ax3.set_xlabel("Condition (%)")

plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.show()
plt.tight_layout()

#plot noise level
np_unc_result_noise = np.array(unc_result_noise)
np_unc_result_drift = np.array(unc_result_drift)


#Plot single row for drifts
#set plot sensor id here to show the sensitivity of metrics toward injected drifts
figsize = (6,3)
dpi = 250
fig, axes = plt.subplots(1,3,figsize=figsize,dpi=dpi)
sensor_id = 10
# x_tick_labels_eff = ('0', '1', '2', '3', '4', '5')
x_tick_labels_eff = ('0', '5', '10', '15', '20', '25')
axes[0].boxplot(recon_drift[sensor_id],showfliers=show_outliers)
axes[0].set_xticklabels(x_tick_labels_eff)
axes[0].set_ylabel("Reconstruction loss")
axes[0].set_xlabel("Injected Drift (%)")

axes[1].boxplot(np_unc_result_drift[sensor_id,:,0,:].tolist(),showfliers=show_outliers)
axes[1].set_xticklabels(x_tick_labels_eff)
axes[1].set_ylabel("Epistemic uncertainty")
axes[1].set_xlabel("Injected Drift (%)")

axes[2].boxplot(np_unc_result_drift[sensor_id,:,1,:].tolist(),showfliers=show_outliers)
axes[2].set_xticklabels(x_tick_labels_eff)
axes[2].set_ylabel("Aleatoric uncertainty")
axes[2].set_xlabel("Injected Drift (%)")
plt.show()
plt.tight_layout()

#Plot single row for noise
#set plot sensor id here to show the sensitivity of metrics toward injected noise

fig, axes = plt.subplots(1,3,figsize=figsize,dpi=dpi)
sensor_id = 10
x_tick_labels_eff = ('0', '5', '10', '15', '20', '25')
axes[0].boxplot(recon_noise[sensor_id],showfliers=show_outliers)
axes[0].set_xticklabels(x_tick_labels_eff)
axes[0].set_ylabel("Reconstruction loss")
axes[0].set_xlabel("Injected Noise (%)")

axes[1].boxplot(np_unc_result_noise[sensor_id,:,0,:].tolist(),showfliers=show_outliers)
axes[1].set_xticklabels(x_tick_labels_eff)
axes[1].set_ylabel("Epistemic uncertainty")
axes[1].set_xlabel("Injected Noise (%)")

axes[2].boxplot(np_unc_result_noise[sensor_id,:,1,:].tolist(),showfliers=show_outliers)
axes[2].set_xticklabels(x_tick_labels_eff)
axes[2].set_ylabel("Aleatoric uncertainty")
axes[2].set_xlabel("Injected Noise (%)")
plt.show()
plt.tight_layout()

#plot cov. matrix
if model_name == "full_cov":
    plot_sample_index = 20
    y_cov_mu_test = result_test['y_cov'][0][0]
    y_cov_mu_ood_0 = result_ood['y_cov'][0][index_ood[0][plot_sample_index]]
    y_cov_mu_ood_1 = result_ood['y_cov'][0][index_ood[1][plot_sample_index]]

    y_cov_list = [y_cov_mu_test,y_cov_mu_ood_0,y_cov_mu_ood_1]
    plot_name_list = ["y_cov_mu_test","y_cov_mu_ood_0","y_cov_mu_ood_1"]

#plot for increasing noise
if model_name == "full_cov":
    plot_sample_index = 0

    y_cov_mu_noise = [noise_data['y_cov'][0][0] for noise_data in result_noise]
    y_cov_mu_drift = [drift_data['y_cov'][0][0] for drift_data in result_drift]

    plot_name_noise_list = [str(i)+"_Noise" for i in range(len(y_cov_mu_noise))]
    plot_name_drift_list = [str(i)+"_Drift" for i in range(len(y_cov_mu_drift))]


    matrix_list = [y_cov_mu_test,y_cov_mu_ood_1,y_cov_mu_noise[-1],y_cov_mu_drift[-1]]
    vmin, vmax = np.array(matrix_list).min(),np.array(matrix_list).max()
    num_plots = len(matrix_list)

    fig, axes = plt.subplots(2,2)
    axes = axes.reshape(-1)
    sup_titles = ["a) Healthy condition","b) Near breakdown","c) Injected Noise (5%)","d) Injected Drift (5 bar)"]
    for id_,cov_mat in enumerate(matrix_list):
        im = axes[id_].imshow(matrix_list[id_],cmap='viridis')
        axes[id_].set_title(sup_titles[id_])
        plt.colorbar(im, ax=axes[id_])

#plot colored samples
result_plot = result_ood
num_colored_samples = 30
index_sample = np.arange(0,num_colored_samples)

epi_sample = np.transpose(result_plot['y_pred'][1][index_sample])
recon_sample = np.transpose((result_plot['y_pred'][0][index_sample]-x_test_noise[1][0][index_sample])**1)
recon_sample = np.abs(recon_sample)
alea_sample = np.transpose(result_plot['y_cov'][0][index_sample])

num_sensors = 17
epi_sample_reshuffled = np.zeros((num_sensors,num_colored_samples))
recon_sample_reshuffled = np.zeros((num_sensors,num_colored_samples))
alea_sample_reshuffled = np.zeros((num_sensors,num_colored_samples))

feature_sensor_index =np.arange(60)*17
feature_sensor_index_list = []

#create index
for i in range(17):
    start_index = i*60
    end_index = (i+1)*60
    feature_sensor_index_list.append(copy.copy(feature_sensor_index)+i)
    epi_sample_reshuffled[i,:]=epi_sample[feature_sensor_index_list[-1],:].mean(0)
    recon_sample_reshuffled[i,:]=recon_sample[feature_sensor_index_list[-1],:].mean(0)
    alea_sample_reshuffled[i,:]=alea_sample[feature_sensor_index_list[-1],:].mean(0)

#3D Coordinate plot
fig = plt.figure(dpi=250)
ax = fig.add_subplot(111, projection='3d')

marker_alpha = 0.6
marker_size = 15
z_scaler = 1e-3
for plot_ood_index in [0,1,2]:
    ax.scatter(recon_loss_plot[plot_ood_index], epi_unc_plot[plot_ood_index], alea_unc_plot[plot_ood_index]*z_scaler, alpha=marker_alpha, s=marker_size)

legend_list = ["Healthy", "Cooler Cond. (20%)", "Cooler Cond. (5%)"]

for plot_drift_level_index in [1,2,3,4]:
    ax.scatter(recon_drift[sensor_id][plot_drift_level_index], np_unc_result_drift[sensor_id,plot_drift_level_index,0,:], np_unc_result_drift[sensor_id,plot_drift_level_index,1,:]*z_scaler, marker='^', alpha=marker_alpha,s=marker_size)
    legend_list.append("Inj. Drift ("+str(x_tick_labels_eff[plot_drift_level_index])+"%)")

for plot_noise_level_index in [1,2,3,4]:
    ax.scatter(recon_noise[sensor_id][plot_noise_level_index], np_unc_result_noise[sensor_id,plot_noise_level_index,0,:], np_unc_result_noise[sensor_id,plot_noise_level_index,1,:]*z_scaler, marker='x', alpha=marker_alpha,s=marker_size)
    legend_list.append("Inj. Noise ("+str(x_tick_labels_eff[plot_noise_level_index])+"%)")

ax.set_xlabel('Reconstruction Loss')
ax.set_ylabel('Epistemic Uncertainty')
ax.set_zlabel('Aleatoric Uncertainty')

ax.legend(legend_list, prop={'size': 8})
ax.text2D(0.06, 0.81, '$\\times 10^{3}$', transform=ax.transAxes)

#========Unsupervised clustering==========
# unsupervised_data = []
# unsupervised_labels_true = []
# z_scaler = 1e-6
# for plot_ood_index,label_index in zip([0,1,2],[0,1,2]):
#     real_drifts = np.array([recon_loss_plot[plot_ood_index],
#                             epi_unc_plot[plot_ood_index],
#                             alea_unc_plot[plot_ood_index]*z_scaler]
#                            )
#     unsupervised_data.append(real_drifts)
#     unsupervised_labels_true.append(np.ones_like(recon_loss_plot[plot_ood_index]) * label_index)
# for plot_drift_level_index,label_index in zip([1,2,3,4],[3,4,5,6]):
#     injected_drifts = np.array([recon_drift[sensor_id][plot_drift_level_index],
#                                 np_unc_result_drift[sensor_id,plot_drift_level_index,0,:],
#                                 np_unc_result_drift[sensor_id,plot_drift_level_index,1,:]*z_scaler])
#     unsupervised_data.append(injected_drifts)
#     unsupervised_labels_true.append(np.ones_like(recon_drift[sensor_id][plot_drift_level_index]) * label_index)
# for plot_noise_level_index,label_index in zip([1,2,3,4],[7,8,9,10]):
#     injected_noise = np.array([recon_noise[sensor_id][plot_noise_level_index],
#                                np_unc_result_noise[sensor_id,plot_noise_level_index,0,:],
#                                np_unc_result_noise[sensor_id,plot_noise_level_index,1,:]*z_scaler])
#     unsupervised_data.append(injected_noise)
#     unsupervised_labels_true.append(np.ones_like(recon_noise[sensor_id][plot_noise_level_index]) * label_index)
#
#
# unsupervised_data = np.concatenate(unsupervised_data,axis=1)
# unsupervised_data = np.moveaxis(unsupervised_data, 0,1)
# unsupervised_labels_true = np.concatenate(unsupervised_labels_true).astype("int")
#
# from sklearn import metrics
# from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
# from sklearn.preprocessing import StandardScaler
# #split 70-30?
#
# def dbscan_predict(model, X):
#
#     nr_samples = X.shape[0]
#
#     y_new = np.ones(shape=nr_samples, dtype=int) * -1
#
#     for i in range(nr_samples):
#         diff = model.components_ - X[i, :]  # NumPy broadcasting
#
#         dist = np.linalg.norm(diff, axis=1)  # Euclidean distance
#
#         shortest_dist_idx = np.argmin(dist)
#
#         if dist[shortest_dist_idx] < model.eps:
#             y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]
#
#     return y_new
#
# #train unsupervised k-means
# sil_scores = []
# # unsupervised_train = unsupervised_data[:,[0,1]]
# # unsupervised_train = unsupervised_data[:,[1,2]]
# # unsupervised_train = unsupervised_data[:,[0,1,2]]
# unsupervised_train = unsupervised_data[:,[0]]
# n_clusters_range= np.arange(3,20)
#
# # for n_clusters in n_clusters_range:
# #     kmeans_model = KMeans(n_clusters=n_clusters, random_state=10).fit(unsupervised_train)
# #     trained_labels = kmeans_model.labels_
# #     sil_score = metrics.silhouette_score(unsupervised_train, trained_labels, metric='euclidean')
# #     print(sil_score)
# #     sil_scores.append(sil_score)
# # best_n_clusters = n_clusters_range[np.argmax(sil_scores)]
# # print("BEST N CLUSTERS:{}".format(best_n_clusters))
# # print("SILHOUTTE SCORE:{}".format(np.max(sil_scores)))
#
# best_n_clusters = 11
# # unsupervised_model = KMeans(n_clusters=best_n_clusters).fit(unsupervised_train)
# # unsupervised_model = AgglomerativeClustering().fit(unsupervised_train)
# unsupervised_model = DBSCAN(eps=2, min_samples=5).fit(unsupervised_train)
#
# #predict using k-means
# # unsupervised_labels_pred = unsupervised_model.predict(unsupervised_train)
# # unsupervised_labels_pred = AgglomerativeClustering(n_clusters=11).fit_predict(unsupervised_train)
# unsupervised_labels_pred = dbscan_predict(unsupervised_model,unsupervised_train)
# nmi_score = metrics.normalized_mutual_info_score(unsupervised_labels_true, unsupervised_labels_pred)
# fowlkes_mallows_score = metrics.fowlkes_mallows_score(unsupervised_labels_true, unsupervised_labels_pred)
# ajrand_score = metrics.adjusted_rand_score(unsupervised_labels_true, unsupervised_labels_pred)
# sil_score = metrics.silhouette_score(unsupervised_train, unsupervised_labels_pred, metric='euclidean')
# print(nmi_score)
# print(ajrand_score)
# print(fowlkes_mallows_score)
# print(sil_score)
#
# fig = plt.figure(dpi=250)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(unsupervised_data[:,0],unsupervised_data[:,1],unsupervised_data[:,2], c=unsupervised_labels_pred)
#
#
#
#
# #==============ADDITIONAL PLOTS===============
# fig, axes = plt.subplots(total_sensors,3)
# for sensor_id in range(total_sensors):
#     x_tick_labels_eff = ('0', '5', '10', '15', '20', '25')
#     axes[sensor_id,0].boxplot(recon_noise[sensor_id],showfliers=show_outliers)
#     axes[sensor_id,0].set_xticklabels(x_tick_labels_eff)
#     axes[sensor_id,0].set_ylabel(sensor_names[sensor_id].upper())
#
#     axes[sensor_id,1].boxplot(np_unc_result_noise[sensor_id,:,0,:].tolist(),showfliers=show_outliers)
#     axes[sensor_id,1].set_xticklabels(x_tick_labels_eff)
#     axes[sensor_id,2].boxplot(np_unc_result_noise[sensor_id,:,0,:].tolist(),showfliers=show_outliers)
#     axes[sensor_id,2].set_xticklabels(x_tick_labels_eff)
#
# fig, axes = plt.subplots(total_sensors,3)
# for sensor_id in range(total_sensors):
#     x_tick_labels_eff = ('0', '5', '10', '15', '20', '25')
#     axes[sensor_id,0].boxplot(recon_drift[sensor_id],showfliers=show_outliers)
#     axes[sensor_id,0].set_xticklabels(x_tick_labels_eff)
#     axes[sensor_id,0].set_ylabel(sensor_names[sensor_id].upper())
#
#     axes[sensor_id,1].boxplot(np_unc_result_drift[sensor_id,:,0,:].tolist(),showfliers=show_outliers)
#     axes[sensor_id,1].set_xticklabels(x_tick_labels_eff)
#
#     axes[sensor_id,2].boxplot(np_unc_result_drift[sensor_id,:,1,:].tolist(),showfliers=show_outliers)
#     axes[sensor_id,2].set_xticklabels(x_tick_labels_eff)
#
#

#=======PLOT LATENT VARIABLE===========
# from sklearn.decomposition import PCA
#
# sensor_index = 10
# alea_index = 0
# epi_index = 1
# alea_key = 'y_cov'
# severity_index = 2
#
# squared = 1
#
# epi_sample_noise = (result_noise[sensor_index][severity_index]['y_pred'][epi_index][index_sample])**squared
# epi_sample_drift = (result_drift[sensor_index][severity_index]['y_pred'][epi_index][index_sample])**squared
# epi_sample_ood = (result_ood['y_pred'][epi_index][index_sample])**squared
# epi_sample_test = (result_test['y_pred'][epi_index][index_sample])**squared
#
# alea_sample_noise = (result_noise[sensor_index][severity_index][alea_key][alea_index][index_sample])**squared
# alea_sample_drift = (result_drift[sensor_index][severity_index][alea_key][alea_index][index_sample])**squared
# alea_sample_ood = (result_ood[alea_key][alea_index][index_sample])**squared
# alea_sample_test = (result_test[alea_key][alea_index][index_sample])**squared
#
# alea_plot_list = [alea_sample_test,alea_sample_ood,alea_sample_drift,alea_sample_noise]
# epi_plot_list = [epi_sample_test,epi_sample_ood,epi_sample_drift,epi_sample_noise]
# combined_plot_list = [alea_sample+epi_sample for alea_sample,epi_sample in zip(alea_plot_list,epi_plot_list)]
# pca = PCA(n_components=2)
#
# def get_stacked_labels(plot_list):
#     sample_test = plot_list[0]
#     labels = np.array([np.ones(sample_test.shape[0])*id for id in range(len(plot_list))]).reshape(-1)
#     stacked = np.stack(plot_list).reshape(-1,sample_test.shape[-1])
#     stacked_pc = pca.fit_transform(stacked)
#
#     return stacked_pc,labels
#
# fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(5,10),dpi=150)
# alea_stacked_pc, alea_pc_labels = get_stacked_labels(alea_plot_list)
# epi_stacked_pc, epi_pc_labels = get_stacked_labels(epi_plot_list)
# total_stacked_pc, total_pc_labels = get_stacked_labels(combined_plot_list)
#
# scat1=ax1.scatter(alea_stacked_pc[:,0], alea_stacked_pc[:,1],c=alea_pc_labels)
# ax1.legend(*scat1.legend_elements())
# scat2=ax2.scatter(epi_stacked_pc[:,0], epi_stacked_pc[:,1],c=epi_pc_labels)
# ax2.legend(*scat2.legend_elements())
# scat3=ax3.scatter(total_stacked_pc[:,0], total_stacked_pc[:,1],c=total_pc_labels)
# ax3.legend(*scat3.legend_elements())
#
# #plot latent_z
# sensor_index = -10
# latent_z_test_mu = result_test['latent_z'][0][index_sample]
# latent_z_test_std = result_test['latent_z'][1][index_sample]
# latent_z_ood_mu = result_ood['latent_z'][0][index_sample]
# latent_z_ood_std = result_ood['latent_z'][1][index_sample]
# latent_z_noise_mu = result_noise[sensor_index][severity_index]['latent_z'][0][index_sample]
# latent_z_noise_std = result_noise[sensor_index][severity_index]['latent_z'][1][index_sample]
# latent_z_drift_mu = result_drift[sensor_index][severity_index]['latent_z'][0][index_sample]
# latent_z_drift_std = result_drift[sensor_index][severity_index]['latent_z'][1][index_sample]
#
# def plot_noise_drift_severity(sensor_index,severity_index, mode="mu"):
#     if mode == "mu":
#         latent_z_noise = result_noise[sensor_index][severity_index]['latent_z'][0][index_sample]
#         latent_z_drift = result_drift[sensor_index][severity_index]['latent_z'][0][index_sample]
#     else:
#         latent_z_noise= result_noise[sensor_index][severity_index]['latent_z'][1][index_sample]
#         latent_z_drift = result_drift[sensor_index][severity_index]['latent_z'][1][index_sample]
#
#     scatter_shape = latent_z_noise[:,0].shape
#     plt.scatter(latent_z_noise[:,0],latent_z_noise[:,1], alpha=0.2, c =(1*np.ones(scatter_shape)).astype(int),cmap='summer')
#     plt.scatter(latent_z_drift[:,0],latent_z_drift[:,1], alpha=0.2, c =(1*np.ones(scatter_shape)).astype(int),cmap='viridis')
#
# plt.figure()
# plot_alpha = 0.2
# plt.scatter(latent_z_test_mu[:,0],latent_z_test_mu[:,1])
# plt.scatter(latent_z_ood_mu[:,0],latent_z_ood_mu[:,1])
# sensor_index = -11
# for i in range(1,4):
#     plot_noise_drift_severity(sensor_index=sensor_index,severity_index=i,mode="mu")
# plt.title("Z-MU SENSOR "+str(sensor_index))
#
# plt.figure()
# plot_alpha = 0.2
# plt.scatter(latent_z_test_std[:,0],latent_z_test_std[:,1])
# plt.scatter(latent_z_ood_std[:,0],latent_z_ood_std[:,1])
# sensor_index = -11
# for i in range(1,4):
#     plot_noise_drift_severity(sensor_index=sensor_index,severity_index=i,mode="std")
# plt.title("Z-STD SENSOR "+str(sensor_index))
#
# for i in range(17):
#     sensor_index = i
#     x_test_diff = (x_test_noise[sensor_index][0]-x_test_noise[sensor_index][2]).sum()
#     print("NOISE:"+str(i)+str(x_test_diff))
# print("--------------")
# for i in range(17):
#     sensor_index = i
#     x_test_diff = (x_test_drift[sensor_index][0]-x_test_drift[sensor_index][2]).sum()
#     print("DRIFT:"+str(i)+str(x_test_diff))
#
# x_test_diff = (x_test_noise[sensor_index][0]-x_test_noise[sensor_index][3]).sum()
# feature_index=-10
# plt.figure()
# plt.hist(x_test_noise[sensor_index][0][:,feature_index],alpha=0.25)
# plt.hist(x_test_noise[sensor_index][-1][:,feature_index],alpha=0.25)
# plt.hist(x_test_drift[sensor_index][-1][:,feature_index],alpha=0.25)
# plt.legend(["HEALTHY","NOISE","DRIFT"])
#
#
#
