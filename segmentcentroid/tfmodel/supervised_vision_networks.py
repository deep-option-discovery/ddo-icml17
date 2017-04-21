import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import string

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

# Create AlexNet model
def convNet(x, output_dims):

    code = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

    net = slim.conv2d(x, 16, [3, 3], 4, padding='VALID', scope='conv1'+code)
    net = slim.max_pool2d(net, [3, 3], 2, scope='pool1'+code)
    net = slim.conv2d(net, 64, [5, 5], scope='conv2'+code)
    net = slim.max_pool2d(net, [3, 3], 2, scope='pool2'+code)
    net = slim.conv2d(net, 64, [3, 3], scope='conv3'+code)
    net = slim.max_pool2d(net, [3, 3], 2, scope='pool5'+code)
    net = slim.conv2d(net, 384, [1, 1], scope='fc7'+code)
    net = slim.dropout(net, 0.5, scope='dropout6'+code)
    net = slim.conv2d(net, output_dims, [1, 1], scope='fc8'+code)
    net = slim.dropout(net, 0.5, scope='dropout7'+code)
    net = slim.flatten(net)
    
    output = slim.fully_connected(net, output_dims, 
                               activation_fn=None,
                               normalizer_fn=None,
                               biases_initializer=tf.zeros_initializer,
                               scope='fc8r'+code)


    return output



def convNetAR1(sdims, adim, dropout=0.2, variance=10000):
    """
    This alexnet for regression

    Positional arguments:
    sdim -- int dimensionality of the state-space
    adim -- int dimensionality of the action-space
    variance -- float scaling for the probability calculation
    """

    sarraydims = [s for s in sdims]
    sarraydims.insert(0, None)

    print(sarraydims)


    x = tf.placeholder(tf.float32, shape=sarraydims)

    a = tf.placeholder(tf.float32, shape=[None, adim])

    weight = tf.placeholder(tf.float32, shape=[None, 1])

    
    output = convNet(x, adim)

    logprob = tf.reduce_sum((output-a)**2, 1)

    y = tf.exp(-logprob/variance)

    wlogprob = weight*logprob
        
    return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': False}

def convNetAC1(sdims, adim, dropout=0.2):
    """
    This alexnet for classification

    Positional arguments:
    sdim -- int dimensionality of the state-space
    adim -- int dimensionality of the action-space
    variance -- float scaling for the probability calculation

    """

    sarraydims = [s for s in sdims]
    sarraydims.insert(0, None)

    x = tf.placeholder(tf.float32, shape=sarraydims)

    a = tf.placeholder(tf.float32, shape=[None, adim])

    weight = tf.placeholder(tf.float32, shape=[None, 1])
    
    output = convNet(x, adim)

    y = tf.nn.softmax(output)

    logprob = tf.nn.softmax_cross_entropy_with_logits(output, a)

    wlogprob = weight*logprob
        
    return {'state': x, 
                'action': a, 
                'weight': weight,
                'prob': y, 
                'lprob': logprob,
                'wlprob': wlogprob,
                'discrete': True}
