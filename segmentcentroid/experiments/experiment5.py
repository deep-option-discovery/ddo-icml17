#!/usr/bin/env python

from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.tfmodel.GridWorldModel import GridWorldModel
from segmentcentroid.planner.jigsaws_loader import JigsawsPlanner
from segmentcentroid.tfmodel.JHUJigSawsModel import JHUJigSawsModel
from segmentcentroid.planner.traj_utils import *

import numpy as np
import copy

import tensorflow as tf


def runPolicies(demonstrations=10,
                directory='/Users/sanjayk/Downloads/Knot_Tying/kinematics/AllGestures/',
                super_iterations=200,
                sub_iterations=1,
                learning_rate=1e-2):

    j = JigsawsPlanner(directory)

    full_traj = []
    for i in range(0,demonstrations):
        full_traj.append(j.plan())

    """
    j = JigsawsPlanner('/Users/sanjayk/Downloads/Suturing/kinematics/AllGestures/')

    for i in range(0,demonstrations):
        full_traj.append(j.plan())

    j = JigsawsPlanner('/Users/sanjayk/Downloads/Needle_Passing/kinematics/AllGestures/')

    for i in range(0,demonstrations):
        full_traj.append(j.plan())
    """
    

    m  = JHUJigSawsModel(3)

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    m.train(opt, full_traj, super_iterations, sub_iterations)

    j.visualizePlans(full_traj, m, filename="resources/results/exp5-trajs7.png")




