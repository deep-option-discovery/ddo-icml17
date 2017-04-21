from .TFSeparableModel import TFSeparableModel
from .supervised_networks import *
import tensorflow as tf
import numpy as np

class JHUJigSawsModel(TFSeparableModel):
    
    """
    This class defines the abstract class for a tensorflow model for the primitives.
    """

    def __init__(self,  
                 k,
                 statedim=(37,1), 
                 actiondim=(8,1),
                 hidden_layer=64,
                 variance=10000):

        self.hidden_layer = hidden_layer
        self.variance = variance
        
        super(JHUJigSawsModel, self).__init__(statedim, actiondim, k, [0,1],'chain')


    def createPolicyNetwork(self):

        return affine(self.statedim[0], self.actiondim[0], self.variance)  

        #return continuousTwoLayerReLU(self.statedim[0],
                                      #self.actiondim[0],
                                      #self.variance) 

    def createTransitionNetwork(self):

        #return logisticRegression(self.statedim[0], 2)
        return  multiLayerPerceptron(self.statedim[0],
                                    2,
                                    self.hidden_layer)




