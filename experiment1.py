#!/usr/bin/env python

from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.tfmodel.GridWorldModel import GridWorldModel
from segmentcentroid.planner.value_iteration import ValueIterationPlanner
from segmentcentroid.planner.traj_utils import *

import numpy as np
import copy
import pdb
import time 

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import tensorflow as tf

#pdb.set_trace()
def inRoom1(state):
	return (state[1] <= 3) 

def inRoom2(state):
	return (state[1] > 4) 

'''
What does runPolies do? 
First we define the number of demonstration trajectories. 

m=GridWorldModel
	This class defines an abstract environment,
	all environments derive from this class

Env Map
1 1 1 1 1 1 1 1 1 
1 2 0 0 1 0 0 0 1 
1 0 0 0 1 0 0 0 1 
1 0 0 0 1 0 0 0 1 
1 0 0 0 1 0 0 0 1 
1 0 0 0 0 0 0 0 1 
1 0 0 0 1 0 0 3 1 
1 1 1 1 1 1 1 1 1 

For all the 20 demonstrations, 
each demonstration, print the start state and the end state.along with the trajectory number. 


'''
def runPolicies(demonstrations=20,
		super_iterations=1000,#10000
		sub_iterations=0,
		learning_rate=10,
		env_noise=0.3):

	m  = GridWorldModel(4, statedim=(8,9))

	   
	MAP_NAME = 'resources/GridWorldMaps/experiment1.txt'
	gmap = np.loadtxt(MAP_NAME, dtype=np.uint8)
	full_traj = []
	vis_traj = []

	for i in range(0,demonstrations):
		print("Traj",i)
		g = GridWorldEnv(copy.copy(gmap), noise=env_noise)
		# print("Initialized")

		g.generateRandomStartGoal()	
		start = np.argwhere(g.map == g.START)[0]
		goal = np.argwhere(g.map == g.GOAL)[0]
		#generate trajectories start in same room and end different room
		while not ((inRoom1(start) and inRoom2(goal))  or\
				   (inRoom2(start) and inRoom1(goal))):
			# print(inr)
			g.generateRandomStartGoal()	
			start = np.argwhere(g.map == g.START)[0]
			goal = np.argwhere(g.map == g.GOAL)[0]


		print(np.argwhere(g.map == g.START), np.argwhere(g.map == g.GOAL))

		v = ValueIterationPlanner(g)
		traj = v.plan(max_depth=100)

		print (len(traj), 'length of the trajectory')
		#this is length depends on the start, and goal state and the planner output. 
		
		new_traj = []
		for t in traj:

			# now for the length of the trajectory it took to get there, iterate over each step
			a = np.zeros(shape=(4,1))

			s = np.zeros(shape=(8,9))

			a[t[1]] = 1

			s[t[0][0],t[0][1]] = 1
			#s[2:4,0] = np.argwhere(g.map == g.START)[0]
			#s[4:6,0] = np.argwhere(g.map == g.GOAL)[0]

			new_traj.append((s,a))

		full_traj.append(new_traj)
		vis_traj.extend(new_traj)
	print (full_traj)
	#raise ValueError("")

	#g.visualizePlan(vis_traj,blank=True, filename="resources/results/exp1-trajs.png")


	m.sess.run(tf.initialize_all_variables())

	with tf.variable_scope("optimizer"):
		opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		#define he optimizer,  put the full trajectorty, 1000, 0 
		m.train(opt, full_traj, super_iterations, sub_iterations)

	actions = np.eye(4)


	g = GridWorldEnv(copy.copy(gmap), noise=0.0)
	
	g.generateRandomStartGoal()

	for i in range(m.k):
		states = g.getAllStates()
		print ('\n',states,'\n','this is all states' , m.k)
		policy_hash = {}
		trans_hash = {}

		for s in states:

			t = np.zeros(shape=(8,9))

			t[s[0],s[1]] = 1
			#t[2:4,0] = np.argwhere(g.map == g.START)[0]
			#t[4:6,0] = np.argwhere(g.map == g.GOAL)[0]

			#np.ravel returns the elements of the combined set of elements. 
			l = [ np.ravel(m.evalpi(i, [(t, actions[j,:])] ))  for j in g.possibleActions(s)]
			print ('\n',l, 'l' ,g.possibleActions(s),'possible actioins')

			if len(l) == 0:
				continue

			#print(i, s,l, m.evalpsi(i,ns))
			action = g.possibleActions(s)[np.argmax(l)]

			policy_hash[s] = action

			#print("Transition: ",m.evalpsi(i, [(t, actions[1,:])]), t)
			trans_hash[s] = np.ravel(m.evalpsi(i, [(t, actions[1,:])]))

		g.visualizePolicy(policy_hash, trans_hash, blank=True, filename="resources/results/exp1-policy"+str(i)+".png")

runPolicies()



