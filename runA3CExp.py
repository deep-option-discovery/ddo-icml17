#!/usr/bin/env python
from segmentcentroid.a3c.driver import train, collect_demonstrations
from segmentcentroid.tfmodel.AtariVisionModel import AtariVisionModel
from segmentcentroid.a3c.augmentedEnv import AugmentedEnv
from segmentcentroid.inference.forwardbackward import ForwardBackward
import tensorflow as tf


from tensorflow.python.client.session import Session
from tensorflow.python.framework.ops import Operation
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.training.saver import Saver
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.ops.variables import Variable
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.framework.dtypes import DType
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.core.framework.op_def_pb2 import OpDef
from tensorflow.python.framework.ops import Graph
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from segmentcentroid.a3c.envs import Diagnostic, DiagnosticsLogger, AtariProcessing
from _frozen_importlib_external import SourceFileLoader
from segmentcentroid.a3c.LSTM import LSTMPolicy
from google.protobuf.internal.python_message import GeneratedProtocolMessageType

import ray
#import module

"""
ray.register_class(AtariVisionModel)
ray.register_class(ForwardBackward)
ray.register_class(Session)
ray.register_class(Operation)
ray.register_class(Tensor)
ray.register_class(Saver)
ray.register_class(AdamOptimizer)
ray.register_class(Variable)
ray.register_class(TensorShapeProto, pickle=True)
ray.register_class(TensorShape)
ray.register_class(DType)
ray.register_class(NodeDef, pickle=True)
ray.register_class(OpDef, pickle=True)
ray.register_class(Graph)
ray.register_class(type, pickle=True)
ray.register_class(GradientDescentOptimizer, pickle=True)
ray.register_class(Diagnostic)
ray.register_class(DiagnosticsLogger)
ray.register_class(AtariProcessing)
ray.register_class(SourceFileLoader)
ray.register_class(LSTMPolicy)
ray.register_class(GeneratedProtocolMessageType, pickle=True)
#ray.register_class(module, pickle=True)
"""

ray.init(num_cpus=2)

env, policy = train(2)
trajs = collect_demonstrations(env, policy)
a = AtariVisionModel(2)

with tf.variable_scope("optimizer2"):
    opt = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    a.sess.run(tf.initialize_all_variables())
    a.train(opt, trajs, 2, 2)

variables = ray.experimental.TensorFlowVariables(a.loss, a.sess)

weights = variables.get_weights()


a2 = AtariVisionModel(2)
a2.sess.run(tf.initialize_all_variables())
#env, policy = train(2, model=weights, k=2)
variables = ray.experimental.TensorFlowVariables(a2.loss, a2.sess)
variables.set_weights(weights)

#print(env.step(1))

