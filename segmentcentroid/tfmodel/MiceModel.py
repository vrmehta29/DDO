from .TFSeparableModel import TFSeparableModel
from .supervised_networks import *

class GridWorldModel(TFSeparableModel):
    
    """
    This class defines the abstract class for a tensorflow model for the primitives.
    """
    #The action dimension for this will have to change?  Hidden layers can stay the same 
    def __init__(self, 
                 k,
                 statedim=(12,2), 
                 actiondim=(4,1), 
                 hidden_layer=8):

        self.hidden_layer = hidden_layer
        
        super(GridWorldModel, self).__init__(statedim, actiondim, k, [0,1], 'cluster')


    def createPolicyNetwork(self):

        #return multiLayerPerceptron(self.statedim[0], self.actiondim[0])
        return gridWorldTabular(12, 2, 4)

    def createTransitionNetwork(self):

        return gridWorldTabular(12, 2, 2)

        

