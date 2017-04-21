from .TFSeparableModel import TFSeparableModel
from .supervised_networks import *

class GridWorldMiddleModel(TFSeparableModel):
    
    """
    This class defines the abstract class for a tensorflow model for the primitives.
    """

    def __init__(self, 
                 k,
                 statedim=(2,1), 
                 actiondim=(4,1), 
                 hidden_layer=8):

        self.hidden_layer = hidden_layer
        
        super(GridWorldMiddleModel, self).__init__(statedim, actiondim, k, [0,1], 'chain')


    def createPolicyNetwork(self):

        #return multiLayerPerceptron(self.statedim[0], self.actiondim[0])
        return gridWorldTabular(10, 20, 4)

    def createTransitionNetwork(self):

        return gridWorldTabular(10, 20, 2)

        

