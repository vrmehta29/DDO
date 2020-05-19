import h5py
import numpy as np 
import matplotlib.pyplot as plt 
filename = '../../Data/pose1.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

print (list(f.keys())[0])

# # Get the data
data = list(f[a_group_key])

data=f['poseest']['points'].value

data=data.astype(float)
data=data[2000:]

l=len(data)



# ######################################################################
# fig = plt.figure()
# ax = plt.axes(xlim=(-4, 4), ylim=(-4, 4))
# #ax = plt.axes(xlim=(0, 400), ylim=(0, 400))
# line, = ax.plot([], [], lw=3)

# def init():
#     line.set_data([], [])
#     return line,
# def animate(i):
#     x = np.asarray(data[i][:][0])/100#np.linspace(0, 4, 1000)
#     y = np.asarray(data[i][:][1])/100#np.sin(2 * np.pi * (x - 0.01 * i))
#     line.set_data(x, y)
#     return line,

# anim = FuncAnimation(fig, animate, init_func=init,frames=200, interval=200)#, blit=True)
# anim.save('sine_wave.gif', writer='imagemagick')

import matplotlib.animation as animation


              



'''
Things to do today: 
center align all the data, generate action frames between the data points. 
The best way to do this is generate a 4 dimensional vector. 
Now we need to generate     

'''
############### centralising the data #################
data_0=data
mean_data=np.mean(data, axis=1)
print (mean_data)
for i in range(len(data)):
    data[i,:,0]-=mean_data[i][0]
    data[i,:,1]-=mean_data[i][1]

################ Generating action data ################
#### 0: up 1: left 2: down 3:right
l=len(data)
COM_action=np.zeros((l,4,1))
print (np.shape(COM_action))
for i in range(l-1):
    distance = mean_data[i+1]-mean_data[i]
    #print (distance[0], distance[1])
    if (np.abs(distance[0])>=np.abs(distance[1])):
        if (distance[0]>0):
            COM_action[i,3,0]=1
        else:
            COM_action[i,1,0]=1
    else:
        if (distance[1]>0):
            COM_action[i,0,0]=1
        else:
            COM_action[i,2,0]=1



'''
We need to divide up the trajectories in sets. 
'''
full_traj=[]
for i in range(l):
    instant=[]
    instant.append(data[i])
    instant.append(COM_action[i])
    full_traj.append(instant)

print (full_traj)

############ plotting the data ####################

def _update_plot(i, fig, scat):
    datai=data[i][:]
    scat.set_offsets(datai)#(([0, i],[50, i],[100, i]))
    print('Frames: %d' %i)

    return scat,

fig =  plt.figure()                

x = [0, 50, 100]
y = [0, 0, 0]

ax = fig.add_subplot(111)
ax.grid(True, linestyle = '-', color = '0.75')
ax.set_xlim([-50,50])
ax.set_ylim([-50,50])

scat = plt.scatter(x, y, c = x)
scat.set_alpha(0.8)

anim = animation.FuncAnimation(fig, _update_plot, fargs = (fig, scat),
                               frames = 1000, interval = 50)
plt.show()  