import tensorflow as tf
import numpy as np
from numpy.random import RandomState
from segmentcentroid.inference.forwardbackward import ForwardBackward
from tensorflow.python.client import timeline
from sklearn.cluster import KMeans

class TFModel(object):
    """
    This class defines the basic data structure for a hierarchical control model. This
    is a wrapper that handles I/O, Checkpointing, and Training Primitives.
    """

    def __init__(self, 
                 statedim, 
                 actiondim, 
                 k,
                 boundary_conditions,
                 prior,
                 checkpoint_file='/tmp/model.bin',
                 checkpoint_freq=10):
        """
        Create a TF model from the parameters

        Positional arguments:
        statedim -- numpy.ndarray defining the shape of the state-space
        actiondim -- numpy.ndarray defining the shape of the action-space
        k -- float defining the number of primitives to learn

        Keyword arguments:
        checkpoint_file -- string filname to store the learned model
        checkpoint_freq -- int iter % checkpoint_freq the learned model is checkpointed
        """

        self.statedim = statedim 
        self.actiondim = actiondim 

        self.k = k

        self.sess = tf.Session()
        self.initialized = False

        self.checkpoint_file = checkpoint_file
        self.checkpoint_freq = checkpoint_freq

        self.initialize()

        self.fb = ForwardBackward(self, boundary_conditions, prior)

        self.saver = tf.train.Saver()

        self.trajectory_cache = {}

        with tf.variable_scope("optimizer"):
            self.opt = tf.train.AdamOptimizer(learning_rate=0.001)
            self.loss, self.optimizer, self.init, self.pivars, self.psivars, self.lossa = self.getOptimizationVariables(self.opt)


    def initialize(self):
        """
        The initialize command is implmented by all subclasses and designed 
        to initialize whatever internal state is needed.
        """

        raise NotImplemented("Must implement an initialize function")


    def restore(self):
        """
        Restores the model from the checkpointed file
        """

        self.saver.restore(self.sess, self.checkpoint_file)

    def save(self):
        """
        Saves the model to the checkpointed file
        """

        self.saver.save(self.sess, self.checkpoint_file)

    def evalpi(self, index, traj):
        """
        Returns the probability of action a at state s for primitive index i

        Positional arguments:
        index -- int index of the required primitive in {0,...,k}
        traj -- a trajectory

        Returns:
        float -- probability
        """

        if index >= self.k:
            raise ValueError("Primitive index is greater than the number of primitives")

        X, A = self.formatTrajectory(traj)

        return np.abs(self._evalpi(index, X, A))


    def evalpsi(self, index, traj):
        """
        Returns the probability of action a at state s

        Positional arguments:
        index -- int index of the required primitive in {0,...,k}
        traj -- a trajectory

        Returns:
        float -- probability
        """

        if index >= self.k:
            raise ValueError("Primitive index is greater than the number of primitives")


        X, _ = self.formatTrajectory(traj)

        return np.abs(self._evalpsi(index, X))


    def _evalpi(self, index, X, A):
        """
        Sub classes must implment this actual execution routine to eval the probability

        Returns:
        float -- probability
        """
        raise NotImplemented("Must implement an _evalpi function")


    def _evalpsi(self, index, X):
        """
        Sub classes must implment this actual execution routine to eval the probability

        Returns:
        float -- probability
        """
        raise NotImplemented("Must implement an _evalpsi function")

    def getLossFunction(self):
        """
        Sub classes must implement a function that returns the loss and trainable variables

        Returns:
        loss -- tensorflow function
        pivars -- variables that handle policies
        psivars -- variables that handle transitions
        """
        raise NotImplemented("Must implement a getLossFunction")


    def dataTransformer(self, trajectory):
        """
        Sub classes can implement a data augmentation class. The default is the identity transform

        Positional arguments: 
        trajectory -- input is a single trajectory

        Returns:
        trajectory
        """
        return trajectory


    """
    ####
    Fitting functions. Below we include functions for fitting the models.
    These are mostly for convenience
    ####
    """

    def sampleBatch(self, X):
        """
        sampleBatch executes the forward backward algorithm and returns
        a single batch of data to train on.

        Positional arguments:
        X -- a list of trajectories. Each trajectory is a list of tuples of states and actions
        dataTransformer -- a data augmentation routine
        """

        #loss, pivars, psivars = self.getLossFunction()

        traj_index = np.random.choice(len(X))
        import datetime
        now  = datetime.datetime.now()
        trajectory = self.trajectory_cache[traj_index]

        #print("Time", datetime.datetime.now()-now)

        now  = datetime.datetime.now()
        weights = self.fb.fit([trajectory])

        #print("Time", datetime.datetime.now()-now)

        feed_dict = {}
        Xm, Am = self.formatTrajectory(trajectory)

        #print(Xm.shape, Am.shape, weights[0][0][:,0].shape, weights[0][1][:,0].shape)


        #prevent stupid shaping errors
        if Xm.shape[0] != weights[0][0].shape[0] or \
           Am.shape[0] != weights[0][1].shape[0]:
            raise ValueError("Error in shapes in np array passed to TF")


        #print("##Q##",weights[0][1][:,0])

        for j in range(self.k):
            feed_dict[self.pivars[j][0]] = Xm
            feed_dict[self.pivars[j][1]] = Am
            feed_dict[self.pivars[j][2]] = np.reshape(weights[0][0][:,j], (Xm.shape[0],1))

            feed_dict[self.psivars[j][0]] = Xm
            feed_dict[self.psivars[j][1]] = self.formatTransitions(weights[0][1][:,j])
            feed_dict[self.psivars[j][2]] = np.reshape(weights[0][0][:,j], (Xm.shape[0],1))

        return feed_dict

    
    def sampleInitializationBatch(self, X, randomBatch, initializationModel):


        #loss, pivars, psivars = self.getLossFunction()

        traj_index = np.random.choice(len(X))

        trajectory = self.trajectory_cache[traj_index]

        weights = (np.zeros((len(X[traj_index]), self.k)), np.ones((len(X[traj_index]), self.k)))

        for i,t in enumerate(trajectory):
            state = t[0].reshape(1,-1)

            index = initializationModel.predict(state)#int(i/ ( len(trajectory)/self.k ))
            weights[0][i, index] = 1
            weights[1][i, index] = 0

        feed_dict = {}
        Xm, Am = self.formatTrajectory(trajectory)

        for j in range(self.k):
            feed_dict[self.pivars[j][0]] = Xm
            feed_dict[self.pivars[j][1]] = Am
            feed_dict[self.pivars[j][2]] = np.reshape(weights[0][:,j], (Xm.shape[0],1))

            feed_dict[self.psivars[j][0]] = Xm
            feed_dict[self.psivars[j][1]] = self.formatTransitions(weights[1][:,j])
            feed_dict[self.psivars[j][2]] = np.reshape(weights[0][:,j], (Xm.shape[0],1))

        return feed_dict

        
    def formatTrajectory(self, 
                         trajectory, 
                         statedim=None, 
                         actiondim=None):
        """
        Internal method that unzips a trajectory into a state and action tuple

        Positional arguments:
        trajectory -- a list of state and action tuples.
        """

        #print("###", statedim, actiondim)

        if statedim == None:
            statedim = self.statedim

        if actiondim == None:
            actiondim = self.actiondim

        sarraydims = [s for s in statedim]
        sarraydims.insert(0, len(trajectory))
        #creates an n+1 d array 

        aarraydims = [a for a in actiondim]
        aarraydims.insert(0, len(trajectory))
        #creates an n+1 d array 

        X = np.zeros(tuple(sarraydims))
        A = np.zeros(tuple(aarraydims))

        for t in range(len(trajectory)):
            #print(t, trajectory[t][0], trajectory[t][0].shape, statedim)
            s = np.reshape(trajectory[t][0], statedim)
            a = np.reshape(trajectory[t][1], actiondim)

            X[t,:] = s
            A[t,:] = a


        #special case for 2d arrays
        if len(statedim) == 2 and \
           statedim[1] == 1:
           X = np.squeeze(X,axis=2)
           #print(X.shape)
        
        if len(actiondim) == 2 and \
           actiondim[1] == 1:
           A = np.squeeze(A,axis=2)
           #print(A.shape)

        return X, A

    def formatTransitions(self, transitions):
        """
        Internal method that turns a transition sequence (array of floats [0,1])
        into an encoded array [1-a, a]
        """

        X = np.zeros((len(transitions),2))
        for t in range(len(transitions)):
            X[t,0] = (1 - transitions[t]) #(1 - transitions[t]) > transitions[t]
            X[t,1] = transitions[t] #transitions[t] > (1 - transitions[t])
        
        return X


    def getOptimizationVariables(self, opt):
        """
        This is an internal method that returns the tensorflow refs
        needed for optimization.

        Positional arguments:
        opt -- a tf.optimizer
        """
        loss, pivars, psivars, lossa = self.getLossFunction()
        train = opt.minimize(loss)

        list_of_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='optimizer')
        init = tf.initialize_variables(list_of_variables)
        return (loss, train, init, pivars, psivars, lossa)


    def startTraining(self, opt):
        """
        This method initializes the training routine

        opt -- is the chosen optimizer to use
        """

        self.gradients = opt.compute_gradients(self.loss)
        self.sess.run(self.init)
        self.initialized = True
        #tf.get_default_graph().finalize()


    def train(self, opt, X, iterations, vqiterations=100, vqbatchsize=25):
        """
        This method trains the model on a dataset weighted by the forward 
        backward algorithm

        Positional arguments:
        opt -- a tf.optimizer
        X -- a list of trajectories
        iterations -- the number of iterations
        vqiterations -- the number of iterations to initialize via vq
        """

        if not self.initialized:

            self.startTraining(opt)

            for i,x in enumerate(X):
                self.trajectory_cache[i] = self.dataTransformer(x)

            #print("abc")

        if vqiterations != 0:
            self.runVectorQuantization(X, vqiterations, vqbatchsize)
            
        for it in range(iterations):

            #if it % self.checkpoint_freq == 0:
                #print("Checkpointing Train", it, self.checkpoint_file)
                #self.save()

            batch = self.sampleBatch(X)

            print("Iteration", it, np.argmax(self.fb.Q,axis=1))

            print("cLoss 1",np.mean(self.sess.run(self.policy_networks[0]['wlprob'], batch)))
            print("cLoss 2" , np.mean(self.sess.run(self.policy_networks[1]['wlprob'], batch)))
            print("tLoss 1",np.mean(self.sess.run(self.transition_networks[0]['wlprob'], batch)))
            print("tLoss 2" , np.mean(self.sess.run(self.transition_networks[1]['wlprob'], batch)))

            #for i in range(0,1000):
            self.sess.run(self.optimizer, batch)

            #print("Time", datetime.datetime.now()-now)
            #print(self.fb.B)
            #    print("Loss",self.sess.run(self.transition_networks[0]['lprob'], batch))

            #gradients_materialized = self.sess.run(self.gradients, batch)

        #return gradients_materialized #Assumption [(t, gradients_materialized[i]) for i,t in enumerate(tf.trainable_variables())]


    def runVectorQuantization(self, X, vqiterations, vqbatchsize):
        """
        This function uses vector quantization to initialize the primitive inference
        """

        state_action_array = []

        for x in X:
            trajectory = self.dataTransformer(x)
            #print([t[0]  for t in trajectory])

            state_action_array.extend([ np.ravel(t[0].reshape(1, -1))  for t in trajectory])
        

        kmeans = KMeans(n_clusters=self.k, init ='k-means++')
        kmeans.fit(state_action_array)

        """
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        p = PCA(n_components=2)
        x = p.fit_transform(state_array)
        plt.scatter(x[:,0], x[:,1])
        plt.show()
        raise ValueError("Break Point")
        """

        for i in range(vqiterations):
            batch = self.sampleInitializationBatch(X, vqbatchsize, kmeans)
            self.sess.run(self.optimizer, batch)
            print("VQ Loss",self.sess.run(self.loss, batch))
            #print("b",self.sess.run(self.lossa[0], batch))
            #print("b1",self.sess.run(self.lossa[0], batch).shape)
            #print("c",self.sess.run(self.lossa[self.k], batch))





        
