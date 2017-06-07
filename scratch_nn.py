import numpy as np
'''
Created on 6 juin 2017

@author: peline
'''

class AbstractBlock(object):
    ''' 
    Abstract class of a block
    a block has multiple input, one output.
    a block must have forward() and backward() method
    '''
    def __init__(self, input_dim):
        self.input_dim = input_dim
        # input data
        self.x = np.zeros(input_dim)
        # output data
        self.y = 0;
        # gradient data
        self.grad_x = np.zeros(input_dim)

    def forward(self, input_data):
        raise NotImplementedError('forward not implemented')

    def backward(self, output_gradient):
        raise NotImplementedError('backward not implemented')

class Neuron(AbstractBlock):   
    '''
    Neuron class computing a weighted sum of its input
    ''' 
    def __init__(self, input_dim):
        super(Neuron, self).__init__(input_dim)    
        # initialize random weight
        self.weight = np.random.rand(input_dim);
        # initialize random bias
        self.bias = np.random.rand(1);
    
    def forward(self, input_data):
        # save input data
        self.x = input_data;
        # compute output
        self.y = np.dot(self.x, self.weight) + self.bias;
        return self.y
    
class Relu(AbstractBlock):
    '''
    Relu activation function block
    '''
    def __init__(self):
        super(Relu, self).__init__(1)    
    
    def forward(self, input_data):
        # save input data
        self.x = input_data
        # compute output
        self.y = max(0.0,self.x)
        return self.y

class AbstractLoss(AbstractBlock):
    '''
    Abstract class of a loss function
    A loss function is a block with setExpectedData() method to set
    the ground truth result of a neural network   
    '''
    def __init__(self, input_dim):
        super(AbstractLoss, self).__init__(input_dim)
        self.expected_data = np.zeros(input_dim)
    
    def setExpectedData(self, expected_data):
        self.expected_data = expected_data   
        
class LossL2(AbstractLoss):
    '''
    A loss function that compute the L2 norm between its input_data and
    ground truth data
    '''
    def __init__(self, input_dim):
        super(LossL2, self).__init__(input_dim)    
    
    def forward(self, input_data):
        # save input data
        self.x = input_data
        # compute L2 distance between input data and expected input data
        self.y = np.sum(np.square(self.x - self.expected_data))
        return self.y   

class Network(object):
    '''
    Class to build/train/run a network made of blocks
    '''
    def __init__(self, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._network = []
        self._last_input_dim = input_dim
        self._loss = None 
        self._last_network_output = np.zeros(output_dim)
    
    def addNeuronlayer(self, NeuronClassType, size):
        layer = {
            'data' : [ NeuronClassType(self._last_input_dim) for i in range(size)],
            'type' : 'neuron'
        }
        self._network.append(layer)
        self._last_input_dim = size        
    
    def addActivationLayer(self, ActivationClassType):
        layer = {
            'data' : [ ActivationClassType() for i in range(self._last_input_dim)],
            'type' : 'activation'
        }
        self._network.append(layer)
        pass
    
    def setLoss(self, LossClass):
        self._loss = LossClass(self._output_dim)
        
    def _forward(self, input_data):
        layer_input = input_data
        
        for layer in self._network:
            # temporary layer ouput
            layer_output = np.zeros(len(layer['data']))
            
            if layer['type'] == 'neuron':
                # feed each block of the layer with the input data
                b_i=0
                for b in layer['data']:
                    layer_output[b_i] = b.forward(layer_input)
                    b_i+=1                   
            elif layer['type'] == 'activation':
                # one to one connection between input data and ouput data elements
                b_i=0
                for b in layer['data']:
                    layer_output[b_i] = b.forward(layer_input[b_i])
                    b_i+=1             
            else:
                raise ValueError('unknow layer type %s, can not run the network' % layer['type'])   
            
            # current ouput is the input of the next layer
            layer_input = layer_output
        
        return layer_output
    
    '''
    Train the network with examples
    (X,Y) training data set (each row is an example)
    '''
    def train(self,X,Y,epoch):
        for e in range(epoch):
            for x,y in zip(X,Y):
                # do forward pass
                network_output = self._forward(x)
                # compute loss
                self._loss.setExpectedData(y)
                l = self._loss.forward(network_output)
                print("loss %f" % l)               
                # back propagate
                # update weight
        
    
    def run(self):
        pass
        
        

def main():
    neuron = Neuron(5)
    x = np.random.rand(5)
    print("random neuron output:")
    print(neuron.forward(x))
    
    loss = LossL2(3)
    print("L2 loss ouput:")
    print(loss.forward([1, 2, 3]))
    
    print("build network model...")
    model = Network(5, 1)
    model.addNeuronlayer(Neuron,5)
    model.addActivationLayer(Relu)
    model.addNeuronlayer(Neuron,3)
    model.addActivationLayer(Relu)
    model.addNeuronlayer(Neuron,1)
    model.setLoss(LossL2)
    
    print("random network forward pass...")
    X = np.matrix([[1.0, 2.0, 3.0, 4.0, 5.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0]])
    Y = np.array([5.0, 0.0])
    print("network ouput :")
    res = model._forward(X[0,:])
    print(res)
    model.train(X,Y,10)
    
    
    
    
    
if __name__ == "__main__":
    main()    

        