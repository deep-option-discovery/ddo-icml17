import tensorflow as tf
import numpy as np
from .TFModel import TFModel
from segmentcentroid.inference.forwardbackward import ForwardBackward
from tensorflow.python.client import timeline


class TFLayeredModel(TFModel):
    """
    This class defines a layered model where the first layer of the 
    model is learned with unsupervised learning and the primitives
    derive features from that layer
    """

    def __init__(self, 
                 actualStateDim,
                 encodedStateDim, 
                 actiondim, 
                 k):
        """
        Create a model from the parameters

        Positional arguments:
        statedim -- numpy.ndarray defining the shape of the state-space
        actiondim -- numpy.ndarray defining the shape of the action-space
        k -- float defining the number of primitives to learn
        """

        self.policy_networks = []
        self.transition_networks = []
        self.encoder = None
        self.actualStateDim = actualStateDim

        super(TFLayeredModel, self).__init__(encodedStateDim, actiondim, k, [0,1], 'chain')



    def getLossFunction(self):

        """
        Returns a loss function that sums over policies and transitions
        """

        loss_array = []

        pi_vars = []
        for i in range(0, self.k):
            loss_array.append(self.policy_networks[i]['wlprob'])
            pi_vars.append((self.policy_networks[i]['state'], 
                            self.policy_networks[i]['action'], 
                            self.policy_networks[i]['weight']))

        psi_vars = []
        for i in range(0, self.k):
            loss_array.append(self.transition_networks[i]['wlprob'])
            psi_vars.append((self.transition_networks[i]['state'], 
                            self.transition_networks[i]['action'], 
                            self.transition_networks[i]['weight']))

        return tf.reduce_sum(loss_array), pi_vars, psi_vars, loss_array



    def initialize(self):
        """
        Initializes the internal state
        """

        with tf.variable_scope("Supervised", reuse=True):
            for i in range(0, self.k):
                self.policy_networks.append(self.createPolicyNetwork())

            for i in range(0, self.k):
                self.transition_networks.append(self.createTransitionNetwork())

        with tf.variable_scope("Unsupervised", reuse=True):
            self.encoder = self.createUnsupervisedNetwork()


    def createUnsupervisedNetwork(self):
        raise NotImplemented("Subclasses must implement a unsupervised network initializer")


    def preloader(self, traj):
        raise NotImplemented("Subclasses must implement a preloader")


    def pretrain(self, 
                 opt, 
                 trajectories,
                 iterations,
                 minibatch=100):

        train = opt.minimize(self.encoder['loss'])

        self.sess.run(tf.initialize_all_variables())

        for i in range(0,iterations):

            print("Pre-Train Iteration", i)
    
            traj_index = np.random.choice(len(trajectories))
            trajectory = self.preloader(trajectories[traj_index])

            Xm, Am = self.formatTrajectory(trajectory, 
                                            statedim=self.actualStateDim, 
                                            actiondim=self.actiondim)

            batch = np.random.choice(np.arange(0,len(trajectory)), size=minibatch)

            feed_dict = {}
            feed_dict[self.encoder['state']] = Xm[batch, :, :, :]
            feed_dict[self.encoder['action']] = Am[batch, :]

            self.sess.run(train, feed_dict)

            print(self.sess.run(self.encoder['loss'], feed_dict ))


    def dataTransformer(self, traj):
        Xm, Am = self.formatTrajectory(self.preloader(traj), 
                                        statedim=self.actualStateDim, 
                                        actiondim=self.actiondim)
        feed_dict = {}
        feed_dict[self.encoder['state']] = Xm
        feed_dict[self.encoder['action']] = Am
        Xmp = self.sess.run(self.encoder['encoded'], feed_dict)

        #print("###",np.argwhere(Xmp>0))
        #import matplotlib.pyplot as plt
        #plt.imshow(Xmp)
        #plt.show()

        return [ (Xmp[i,:] , Am[i,:]) for i,t in enumerate(traj)]


    #returns a probability distribution over actions
    def _evalpi(self, index, s, a):

        feed_dict = {self.policy_networks[index]['state']: s, 
                     self.policy_networks[index]['action']: a}

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        dist = self.sess.run(self.policy_networks[index]['prob'], feed_dict, options=run_options, run_metadata=run_metadata)

        print("pred", dist)

        if self.policy_networks[index]['discrete']:
            return np.sum(np.multiply(dist,a), axis=1)
        
        return dist
            


    #returns a probability distribution over actions
    def _evalpsi(self, index, s):
        feed_dict = {self.transition_networks[index]['state']: s}
        #print(s)
        dist = self.sess.run(self.transition_networks[index]['prob'], feed_dict)

        print(dist, np.sum(dist, axis=1))

        if not self.transition_networks[index]['discrete']:
            raise ValueError("Transition function must be discrete")

        return dist[:,1]





        
