import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


Gamma=0.0005
q=1.6e-19
m=0.067*9e-31
B=10
Ec=(1.0567e-34)*B/m

fig, ax = plt.subplots()

n = 3 #number of lines
x = np.arange(0, 3.6e-3, 1.7e-5)        # x-array, third number is interval here, x is energy
lines = [ax.plot(x, np.e**(-(x-((1.0567e-34)*1*1/m))**2/Gamma**2), zorder=i+3)[0] for i in range(n)]
fills = [ax.fill_between(x,0,(np.e**(-(x-((1.0567e-34)*1*1/m))**2/Gamma**2)), facecolor=lines[i].get_color(), zorder=i+3) for i in range(n)]


def animate(i):
    for d, line in enumerate(lines):
        p=(d+1)/2.
        line.set_ydata(np.e**((-(x-((1.0567e-34)*p*i/m))**2)/Gamma**2))
        fills[d].remove()
        fills[d] = ax.fill_between(x,0,(np.e**(-(x-((1.0567e-34)*p*i/m))**2/Gamma**2)), facecolor=lines[d].get_color(), zorder=d+3)# update the data

    return lines + fills


#Init only required for blitting to give a clean slate.
def init():
    for line in lines:
        line.set_ydata(np.ma.array(x, mask=True))
    return lines

ani = animation.FuncAnimation(fig, animate, np.arange(0, 2.5, .01), init_func=init,
    interval=10, blit=False)
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
#
#ani.save('QHanimati.mp4', writer=writer)
ani.save('dummy2.gif', writer='imagemagick')
plt.show()