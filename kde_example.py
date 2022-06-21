import numpy as np
import matplotlib.pyplot as plt

def gauss_kernel(x):
    return (1/(np.sqrt(np.pi*2)))*np.exp(-0.5*(x**2))

def gauss_kernel(x):
    return (1/(np.sqrt(np.pi*2)))*np.exp(-0.5*(x**2))

def kde(x, x_data, bw_h=1):
    N = len(x_data)
    scaler = 1/(N*bw_h)
    sum_res = np.sum([gauss_kernel((x-x_i)/bw_h) for x_i in x_data])
    return scaler*sum_res

def kde_mse(x, x_data, bw_h=1):
    N = len(x_data)
    sum_res = np.sum([(x-x_i)**2 for x_i in x_data])/N
    return sum_res

# x_data=np.random.randn(2)
# evaluate_space = np.linspace(-4,4,100)
#
# # bw_h = 1.05*np.std(x_data)*(len(x_data)**(-1/5))
# bw_h = 1
# kde_res = np.array([kde(x, x_data, bw_h=bw_h) for x in evaluate_space ])
#
# # plot data
# plt.figure()
# plt.hist(x_data,density=True)
# plt.plot(evaluate_space, kde_res)
# plt.scatter(x_data,x_data*0-0.01, c="red")
