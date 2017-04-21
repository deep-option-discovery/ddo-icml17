import tensorflow as tf
import numpy as np
import random
import string
import tensorflow.contrib.slim as slim

"""
This file is meant to be a model-zoo like file that creates 
networks that can be used in other parts of the pipeline. These
networks are tensorflow references and need to expose the following API.

{ 
'state': <reference to the state placeholder>
'action': <reference to the action placeholder>
'weight': <reference to the weight placeholder>
'prob': <reference to a variable that calculates the probability of action given state>
'wlprob': <reference to a variable that calculates the weighted log probability> 
'discrete': <True/False describing whether the action space should be one-hot encoded>   
}
"""


def continuousTwoLayerReLU(sdim, adim, variance, hidden_layer=64):
    """
    This function creates a regression network that takes states and
    regresses to actions. It is based on a gated relu.

    Positional arguments:
    sdim -- int dimensionality of the state-space
    adim -- int dimensionality of the action-space
    variance -- float scaling for the probability calculation
    
    Keyword arguments:
    hidden_later -- int size of the hidden layer
    """

    x = tf.placeholder(tf.float32, shape=[None, sdim])

    a = tf.placeholder(tf.float32, shape=[None, adim])

    weight = tf.placeholder(tf.float32, shape=[None, 1])

    W_h1 = tf.Variable(tf.random_normal([sdim, hidden_layer ]))

    b_1 = tf.Variable(tf.random_normal([hidden_layer]))

    h1 = tf.concat(1,[tf.nn.relu(tf.matmul(x, W_h1) + b_1), tf.matmul(x, W_h1) + b_1])

    W_out = tf.Variable(tf.random_normal([hidden_layer*2, adim]))

    b_out = tf.Variable(tf.random_normal([adim]))

    output = tf.matmul(h1, W_out) + b_out

    logprob = tf.reduce_sum((output-a)**2, 1)/variance

    y = tf.exp(-logprob)

    wlogprob = tf.multiply(tf.transpose(weight), logprob)
        
    return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'amax': output,
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': False}



def logisticRegression(sdim, adim):

    """
    This function creates a multinomial logistic regression

    Positional arguments:
    sdim -- int dimensionality of the state-space
    adim -- int dimensionality of the action-space
    """

    x = tf.placeholder(tf.float32, shape=[None, sdim])

    a = tf.placeholder(tf.float32, shape=[None, adim])

    weight = tf.placeholder(tf.float32, shape=[None, 1])

    W_h1 = tf.Variable(tf.random_normal([sdim, adim]))
    b_1 = tf.Variable(tf.random_normal([adim]))
        
    logit = tf.matmul(x, W_h1) + b_1

    y = tf.nn.softmax(logit)

    logprob = tf.nn.softmax_cross_entropy_with_logits(logit, a)

    wlogprob = tf.multiply(tf.transpose(weight), logprob)
        
    return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'amax': tf.argmax(y, 1),
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': True}


def multiLayerPerceptron(sdim, 
                         adim, 
                         hidden_layer=64):
    """
    This function creates a classification network that takes states and
    predicts a hot-one encoded action. It is based on a MLP.

    Positional arguments:
    sdim -- int dimensionality of the state-space
    adim -- int dimensionality of the action-space
    
    Keyword arguments:
    hidden_later -- int size of the hidden layer
    """

    x = tf.placeholder(tf.float32, shape=[None, sdim])

    #must be one-hot encoded
    a = tf.placeholder(tf.float32, shape=[None, adim])

    #must be a scalar
    weight = tf.placeholder(tf.float32, shape=[None, 1])

    W_h1 = tf.Variable(tf.random_normal([sdim, hidden_layer]))
    b_1 = tf.Variable(tf.random_normal([hidden_layer]))
    h1 = tf.nn.sigmoid(tf.matmul(x, W_h1) + b_1)

    #h1 = tf.nn.dropout(h1, 0.5)

    W_out = tf.Variable(tf.random_normal([hidden_layer, adim]))
    b_out = tf.Variable(tf.random_normal([adim]))
        
    logit = tf.matmul(h1, W_out) + b_out
    y = tf.nn.softmax(logit)

    logprob = tf.nn.softmax_cross_entropy_with_logits(logit, a)

    wlogprob = tf.multiply(tf.transpose(weight), logprob)
        
    return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'amax': tf.argmax(y, 1),
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': True}


def conv2affine(sdim, adim, variance, _hiddenLayer=32):
    code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

    sarraydims = [s for s in sdim]
    sarraydims.insert(0, None)

    x = tf.placeholder(tf.float32, shape=sarraydims)

    a = tf.placeholder(tf.float32, shape=[None, adim])

    weight = tf.placeholder(tf.float32, shape=[None, 1])

    net = slim.conv2d(x, 64, [11, 11], 4, padding='VALID', scope='conv1'+code)
    #net = slim.conv2d(net, 192, [5, 5], scope='conv2'+code)
    #net = slim.conv2d(net, 384, [3, 3], scope='conv3'+code)

    net = slim.flatten(net)
    W1 = tf.Variable(tf.random_normal([68096, _hiddenLayer]))
    b1 = tf.Variable(tf.random_normal([_hiddenLayer]))
    output = tf.nn.sigmoid(tf.matmul(net, W1) + b1)
    #output= tf.nn.dropout(output, dropout)

    W2 = tf.Variable(tf.random_normal([_hiddenLayer, adim]))
    b2 = tf.Variable(tf.random_normal([adim]))

    output = tf.nn.sigmoid(tf.matmul(output, W2) + b2)

    logprob = tf.reduce_sum((output-a)**2, 1)/variance

    y = tf.exp(-logprob)

    wlogprob = tf.multiply(tf.transpose(weight), logprob)
        
    return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'amax':output,
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': False}

def conv2mlp(sdim, adim, _hiddenLayer=32):
    code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

    sarraydims = [s for s in sdim]
    sarraydims.insert(0, None)

    x = tf.placeholder(tf.float32, shape=sarraydims)

    a = tf.placeholder(tf.float32, shape=[None, adim])

    weight = tf.placeholder(tf.float32, shape=[None, 1])

    net = slim.conv2d(x, 64, [11, 11], 4, padding='VALID', scope='conv1'+code)
    #net = slim.conv2d(net, 192, [5, 5], scope='conv2'+code)
    #net = slim.conv2d(net, 384, [3, 3], scope='conv3'+code)

    net = slim.flatten(net)
    W1 = tf.Variable(tf.random_normal([68096, _hiddenLayer]))
    b1 = tf.Variable(tf.random_normal([_hiddenLayer]))
    output = tf.nn.sigmoid(tf.matmul(net, W1) + b1)
    #output= tf.nn.dropout(output, dropout)

    W2 = tf.Variable(tf.random_normal([_hiddenLayer, adim]))
    b2 = tf.Variable(tf.random_normal([adim]))

    logit = tf.matmul(output, W2) + b2

    y = tf.nn.softmax(logit)

    logprob = tf.nn.softmax_cross_entropy_with_logits(logit, a)

    wlogprob = tf.multiply(tf.transpose(weight), logprob)
        
    return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'amax': tf.argmax(y, 1),
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': True}


def conv2a3c(sdim, adim, _hiddenLayer=32):
    code = ''#.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

    sarraydims = [s for s in sdim]
    sarraydims.insert(0, None)

    x = tf.placeholder(tf.float32, shape=sarraydims)

    a = tf.placeholder(tf.float32, shape=[None, adim])

    weight = tf.placeholder(tf.float32, shape=[None, 1])

    net = slim.conv2d(x, 32, [11, 11], 4, padding='VALID')
    net = slim.conv2d(net, 64, [5, 5])
    net = slim.conv2d(net, 128, [3, 3])

    net = slim.flatten(net)
    W1 = tf.Variable(tf.random_normal([8192, _hiddenLayer]))
    b1 = tf.Variable(tf.random_normal([_hiddenLayer]))
    output = tf.nn.sigmoid(tf.matmul(net, W1) + b1)
    #output= tf.nn.dropout(output, dropout)

    W2 = tf.Variable(tf.random_normal([_hiddenLayer, adim]))
    b2 = tf.Variable(tf.random_normal([adim]))

    logit = tf.matmul(output, W2) + b2

    y = tf.nn.softmax(logit)

    logprob = tf.nn.softmax_cross_entropy_with_logits(logit, a)

    wlogprob = tf.multiply(tf.transpose(weight), logprob)
        
    return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'amax': tf.argmax(y, 1),
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': True}


def affine(sdim, adim, variance):
    """
    This function creates a linear regression network that takes states and
    regresses to actions. It is based on a gated relu.

    Positional arguments:
    sdim -- int dimensionality of the state-space
    adim -- int dimensionality of the action-space
    variance -- float scaling for the probability calculation
    
    """

    x = tf.placeholder(tf.float32, shape=[None, sdim])

    a = tf.placeholder(tf.float32, shape=[None, adim])

    weight = tf.placeholder(tf.float32, shape=[None, 1])

    W_h1 = tf.Variable(tf.random_normal([sdim, adim]))

    b_1 = tf.Variable(tf.random_normal([adim]))

    output = tf.matmul(x, W_h1) + b_1

    logprob = tf.reduce_sum((output-a)**2, 1)/variance

    y = tf.exp(-logprob)

    wlogprob = tf.multiply(tf.transpose(weight), logprob)
        
    return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'amax':output,
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': False}


def gridWorldTabular(xcard, ycard, adim):
    """
    This function creates a linear regression network that takes states and
    regresses to actions. It is based on a gated relu.

    Positional arguments:
    sdim -- int dimensionality of the state-space
    adim -- int dimensionality of the action-space
    variance -- float scaling for the probability calculation
    
    """

    x = tf.placeholder(tf.float32, shape=[None, xcard, ycard])

    a = tf.placeholder(tf.float32, shape=[None, adim])

    weight = tf.placeholder(tf.float32, shape=[None, 1])

    table = tf.Variable(tf.abs(tf.random_normal([xcard, ycard, adim])))

    inputx = tf.tile(tf.reshape(x, [-1, xcard, ycard, 1]), [1, 1, 1, adim])

    collapse = tf.reduce_sum(tf.reduce_sum(tf.multiply(inputx, table), 1), 1)

    normalization = tf.reduce_sum(tf.abs(collapse),1)
    
    actionsP = tf.abs(collapse) / tf.tile(tf.reshape(normalization, [-1, 1]), [1, adim])

    y = tf.reduce_mean(tf.multiply(a, actionsP), 1)

    logprob = -tf.log1p(y)

    wlogprob = tf.multiply(tf.transpose(weight), logprob)
        
    return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'amax': tf.argmax(actionsP, 1),
                'debug': tf.multiply(a, actionsP),
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': False}
    