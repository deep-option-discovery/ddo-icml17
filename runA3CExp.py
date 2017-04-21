#!/usr/bin/env python
from segmentcentroid.a3c.augmentedEnv import AugmentedEnv
from segmentcentroid.inference.forwardbackward import ForwardBackward
import tensorflow as tf

import ray

"""
Hyper-parameters
"""
DOMAIN = "Frostbite-v0"
PARTIAL_TRAINING = 2e6
DDO_LEARNING_RATE = 1e-3
DDO_MAX_ITER = 1e5
DDO_MAX_VQ_ITER = 1e5
DDO_L1_OPTIONS = 2
DDO_L2_OPTIONS = 4
NUMBER_OF_WORKERS = 128

"""
This code initializes the parallelization
"""
ray.init(num_cpus=NUMBER_OF_WORKERS)


"""
This code uses A3C to train an initial policy
"""
from segmentcentroid.a3c.driver import train, collect_demonstrations
env, policy = train(NUMBER_OF_WORKERS, max_steps=PARTIAL_TRAINING)
trajs = collect_demonstrations(env, policy)


"""
This code imports a predefined DDO model and fits the parameters
"""
from segmentcentroid.tfmodel.AtariVisionModel import AtariVisionModel
a = AtariVisionModel(DDO_L1_OPTIONS, DDO_L2_OPTIONS)

with tf.variable_scope("optimizer2"):
    opt = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    a.sess.run(tf.initialize_all_variables())
    a.train(opt, trajs, 2, 2)

#this code gets the weights of all of the optins learned with DDO
variables = ray.experimental.TensorFlowVariables(a.loss, a.sess)
weights = variables.get_weights()


"""
The A3C policy is reinitialized with the model weights and is 
"""
env, policy = train(2, model=weights)



