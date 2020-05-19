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

data=data.astype('float64')
data=data[20000:]

l=len(data)

import matplotlib.animation as animation

############### centralising the data #################
data_0=data
mean_data=np.mean(data, axis=1)
#print (mean_data)
for i in range(len(data)):
    data[i,:,0]-=mean_data[i][0]
    data[i,:,1]-=mean_data[i][1]

################ Generating action data ################
#### 0: up 1: left 2: down 3:right



'''
Modifying the action dimension here, from (l,4,1) to (l,12,1)
'''
l=len(data)
COM_action=np.zeros((l,12,2))
COM_action=COM_action.astype('float64')
#print (np.shape(COM_action))
for i in range(l-1):
    # distance = mean_data[i+1]-mean_data[i]
    # print (distance[0], distance[1])
    # if (np.abs(distance[0])>=np.abs(distance[1])):
    #     if (distance[0]>0):
    #         COM_action[i,3,0]=1
    #     else:
    #         COM_action[i,1,0]=1
    # else:
    #     if (distance[1]>0):
    #         COM_action[i,0,0]=1
    #     else:
    #         COM_action[i,2,0]=1
    COM_action[i]=(data[i+1]-data[i])
    print (np.shape(COM_action[i]))
    
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

