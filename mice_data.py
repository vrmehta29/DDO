#!/usr/bin/env python

#from segmentcentroid.envs.MiceEnv import GridWorldEnv
from segmentcentroid.tfmodel.MiceModel import GridWorldModel
# from segmentcentroid.planner.value_iteration import ValueIterationPlanner
# from segmentcentroid.planner.traj_utils import *

import numpy as np
import copy
import pdb
import time 

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import h5py

############# Importing the data ###############
filename = '../../Data/pose1.h5'
f = h5py.File(filename, 'r')

# List all groups
#print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

#print (list(f.keys())[0])

# # Get the data
data = list(f[a_group_key])
#print (data)
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
#print (mean_data)
for i in range(len(data)):
    data[i,:,0]-=mean_data[i][0]
    data[i,:,1]-=mean_data[i][1]

################ Generating action data ################
#### 0: up 1: left 2: down 3:right
l=len(data)
COM_action=np.zeros((l,4,1))
#print (np.shape(COM_action))
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

print ("\n Done normalizing data, and actions generated \n")

'''
We need to divide up the trajectories in sets. 
'''
full_traj=[]
episode_len=10
no_episode=10
for j in range(no_episode):
	episode=[]
	for i in range(episode_len):
		#instant=[]
		instant=((data[j*episode_len+i], COM_action[j*episode_len+i]))
		#instant.append(COM_action[i])
		episode.append(instant)
	full_traj.append(episode)


print (full_traj)

#np.savetxt( 'arrays_storage/full_mice_traj.txt',full_traj,fmt='%s')

#pdb.set_trace()
# def inRoom1(state):
# 	return (state[1] <= 3) 	

# def inRoom2(state):
# 	return (state[1] > 4) /

# def runPolicies(demonstrations=1,
# 		super_iterations=1000,#10000
# 		sub_iterations=0,
# 		learning_rate=10,
# 		env_noise=0.3):

	# #m  = GridWorldModel(2, statedim=(8,9))
			   
	# #MAP_NAME = 'resources/GridWorldMaps/experiment1.txt'
	# #gmap = np.loadtxt(MAP_NAME, dtype=np.uint8)
	# #print (gmap)
	# full_traj = []
	# vis_traj = []

	# for i in range(0,demonstrations):
		#print("Traj",i)
		#g = GridWorldEnv(copy.copy(gmap), noise=env_noise)
		# print("Initialized")

		#g.generateRandomStartGoal()	
		#start = np.argwhere(g.map == g.START)[0]
		#goal = np.argwhere(g.map == g.GOAL)[0]
		#generate trajectories start in same room and end different room
		# while not ((inRoom1(start) and inRoom2(goal))  or\
		# 		   (inRoom2(start) and inRoom1(goal))):
		# 	# print(inr)
		# 	g.generateRandomStartGoal()	
		# 	start = np.argwhere(g.map == g.START)[0]
		# 	goal = np.argwhere(g.map == g.GOAL)[0]


		#print(np.argwhere(g.map == g.START), np.argwhere(g.map == g.GOAL))

		#v = ValueIterationPlanner(g)
		#traj = v.plan(max_depth=100)
		#np.savetxt( 'arrays_storage/traj.txt',traj,fmt='%s')
		#print (len(traj), 'length of the trajectory')
		#this is length depends on the start, and goal state and the planner output. 
		
		# new_traj = []
		# for t in traj:

		# 	# now for the length of the trajectory it took to get there, iterate over each step
		# 	a = np.zeros(shape=(4,1))

		# 	s = np.zeros(shape=(12,2))

		# 	a[t[1]] = 1

		# 	s[t[0][0],t[0][1]] = 1
		# 	#s[2:4,0] = np.argwhere(g.map == g.START)[0]
		# 	#s[4:6,0] = np.argwhere(g.map == g.GOAL)[0]

		# 	new_traj.append((s,a))

		# full_traj.append(new_traj)
		# vis_traj.extend(new_traj)
	#np.savetxt( 'arrays_storage/full_traj.txt',full_traj,fmt='%s')
demonstrations=1
super_iterations=1000#10000
sub_iterations=0
learning_rate=10


#k=4 in  this case. number of primitive options
m  = GridWorldModel(4, statedim=(12,2))
m.sess.run(tf.initialize_all_variables())

with tf.variable_scope("optimizer"):
	opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	#define he optimizer,  put the full trajectorty, 1000, 0 
	m.train(opt, full_traj, super_iterations, sub_iterations)

'''So how do we generate the visualised options?
We can look at a state, and then apply the  respective options policy from that state.
so how is this done for the gridworld data? It just computes the max of action probabilities over the entire gridworld. 
Instead of doing that we need to provide the same option policy over continues states until it actally terminates. 
How do we do this? 

1. Find a few good states in the state space. 
2. Iterate over the numerb of options and do the sam ething as before
3. till the termination poliy is reached iterate of v evalpi for the same state space
4. Evalpi will need modifications for the same. 
5. But what are the possible action for each state? all actions are possible.
6. What is the termination policy? 


'''
actions = np.eye(4)

policy_hash = {}
trans_hash = {}
len_option=10
for j in range(len_option):
	for i in range(m.k):
		state=data[1000]
		print (state)
		l=[np.ravel(m.evalpi(i,[(state,actions[j,:])])) for j in [0,1,2,3]]
		action = [np.argmax(l)]
		print (action)
		

		

#code for visualising the generated output trajectories

	# actions = np.eye(4)

############ This part of the code is only for visualization. 

	# g = GridWorldEnv(copy.copy(gmap), noise=0.0)
	# #np.savetxt( 'arrays_storage/g.txt',g,fmt='%s')
	
	# g.generateRandomStartGoal()

	# for i in range(m.k):
	# 	states = g.getAllStates()
	# 	np.savetxt( 'arrays_storage/states.txt',states,fmt='%s')

	# 	policy_hash = {}
	# 	trans_hash = {}

	# 	for s in states: # looping through all the states. 

	# 		t = np.zeros(shape=(8,9))

	# 		t[s[0],s[1]] = 1
	# 		#t[2:4,0] = np.argwhere(g.map == g.START)[0]
	# 		#t[4:6,0] = np.argwhere(g.map == g.GOAL)[0]

	# 		#np.ravel returns the elements of the combined set of elements. 
	# 		l = [ np.ravel(m.evalpi(i, [(t, actions[j,:])] ))  for j in g.possibleActions(s)]
	# 		#np.savetxt( 'arrays_storage/l.txt',l,fmt='%s')

	# 		if len(l) == 0:
	# 			continue

	# 		#print(i, s,l, m.evalpsi(i,ns))
	# 		action = g.possibleActions(s)[np.argmax(l)]

	# 		policy_hash[s] = action

	# 		#print("Transition: ",m.evalpsi(i, [(t, actions[1,:])]), t)
	# 		trans_hash[s] = np.ravel(m.evalpsi(i, [(t, actions[1,:])]))
	# 	#np.savetxt( 'arrays_storage/policy_hash.txt',policy_hash,fmt='%s')
	# 	#np.savetxt( 'arrays_storage/trans_hash.txt',trans_hash,fmt='%s')

	# 	g.visualizePolicy(policy_hash, trans_hash, blank=True, filename="resources/results/exp1-policy"+str(i)+".png")

#runPolicies()

