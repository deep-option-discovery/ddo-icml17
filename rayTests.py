#!/usr/bin/env python

from segmentcentroid.tfmodel.GridWorldNNModel import GridWorldNNModel as GridWorldModel
from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.planner.value_iteration import ValueIterationPlanner
import copy
import numpy as np
import ray
import tensorflow as tf

ray.init(num_workers=1)


###ray starts here

@ray.remote
def planner():
    print("-----Planning Demonstration-----")
    MAP_NAME = 'resources/GridWorldMaps/experiment1.txt'
    gmap = np.loadtxt(MAP_NAME, dtype=np.uint8)

    g = GridWorldEnv(copy.copy(gmap), noise=0.1)
    g.generateRandomStartGoal()
    v = ValueIterationPlanner(g)
    traj = v.plan(max_depth=100)
    new_traj = []
    for t in traj:
        a = np.zeros(shape=(4,1))
        s = np.zeros(shape=(2,1))
        a[t[1]] = 1
        s[0:2,0] =  t[0]
        new_traj.append((s,a))

    return new_traj


def gridWorldInit():

    t = tf.Graph()

    with t.as_default():
        m = GridWorldModel(2, statedim=(2,1))
        
        m.sess.run(tf.initialize_all_variables())

        variables = ray.experimental.TensorFlowVariables(m.loss, m.sess)

    return m, m.opt, t, variables


def gridWorldReinit(m):
    return m


ray.env.gridworld = ray.EnvironmentVariable(gridWorldInit, gridWorldReinit)


@ray.remote
def ptrain(weights, dataset):
    m, opt, t, variables = ray.env.gridworld

    variables.set_weights(weights)

    with t.as_default():
        with tf.variable_scope("optimizer"):
            return m.train(opt, dataset, 1, 0)
    
    return None



demonstrations = 100
dataset = ray.get([planner.remote() for i in range(demonstrations)])


m = GridWorldModel(2, statedim=(2,1))
m.sess.run(tf.initialize_all_variables())

with tf.variable_scope("optimizer"):

    opt = m.opt

    #opt = tf.train.AdamOptimizer(learning_rate=0.001)
    
    m.train(opt, dataset, 1, 0)


    driver_gradients = opt.compute_gradients(m.loss)

    variables = ray.experimental.TensorFlowVariables(m.loss, m.sess)

    vlist = [g for (g,v) in driver_gradients]
    
    update = opt.apply_gradients(driver_gradients)


    variables = ray.experimental.TensorFlowVariables(m.loss, m.sess)

    for iter in range(0,1000):

        print("----Iteration---", iter)

        weights = ray.put(variables.get_weights())



        list_of_gradients = ray.get([ptrain.remote(weights, dataset) for i in range(0,1)])



        for l in list_of_gradients:
        #print(l, vlist)
    
            feedDict = { g:l[i][0] for i, g in enumerate(vlist)}
    
        #import IPython
        #IPython.embed()

            m.sess.run(update, feed_dict=feedDict)
        #print("----Loss----", m.sess.run(m.loss, feed_dict=feedDict))






#train()

