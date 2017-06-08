import numpy as np
# matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
'''
Created on 6 juin 2017

@author: peline
'''

class AbstractBlock(object):
    ''' 
    Abstract class of a block
    a block has multiple input, one output.
    a block must have forward() and backward() methods
    update() method is optional
    '''
    def __init__(self, input_dim):
        self.input_dim = input_dim
        # input data
        self.x = np.zeros(input_dim)
        # output data
        self.y = 0
        # gradient data
        self.grad_x = np.zeros(input_dim)

    '''
    forward pass
    input_data is a vector, each element corresponds to a neuron input
    '''
    def forward(self, input_data):
        raise NotImplementedError('forward not implemented')

    '''
    backward pass
    output_gradient is a scalar
    '''
    def backward(self, output_gradient):
        raise NotImplementedError('backward not implemented')

    '''
    trigger weight update
    '''
    def update(self, learn_rate):
        pass

class Neuron(AbstractBlock):
    '''
    Neuron class computing a weighted sum of its input
    '''
    def __init__(self, input_dim):
        super(Neuron, self).__init__(input_dim)
        # initialize random weight
        self.weight = np.random.normal(loc=0.0, scale=0.1, size=(input_dim))
        # initialize bias
        self.bias = 0.0
        # init weight and bias gradient
        self.grad_weight = np.zeros(input_dim)
        self.grad_bias = 0.0

    def forward(self, input_data):
        # save input data
        self.x = input_data
        # compute output
        self.y = np.dot(self.x, self.weight) + self.bias
        return self.y

    def backward(self, output_gradient):
        self.grad_bias = output_gradient
        self.grad_weight = self.x * output_gradient
        self.grad_x = self.weight * output_gradient
        return self.grad_x

    def update(self, learn_rate):
        self.weight += -learn_rate * self.grad_weight
        self.bias += -learn_rate * self.grad_bias

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
        self.y = max(0.0, self.x)
        return self.y

    def backward(self, output_gradient):
        if self.x > 0:
            self.grad_x = output_gradient
        else:
            self.grad_x = 0.0
        return self.grad_x

class AbstractLoss(AbstractBlock):
    '''
    Abstract class of a loss function
    A loss function is a "block" with a setExpectedData() method to set
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

    def backward(self):
        self.grad_x = 2 * (self.x - self.expected_data)
        return self.grad_x

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

    '''
    Add a layer of neuron fully connected to the previous layer
    '''
    def addNeuronlayer(self, NeuronClassType, size):
        layer = {
            'data' : [ NeuronClassType(self._last_input_dim) for i in range(size)],
            'input_dim' : self._last_input_dim,
            'output_dim' : size,
            'type' : 'neuron'
        }
        self._network.append(layer)
        self._last_input_dim = size

    '''
    Add an activation layer (one activation block per block of the previous layer)
    '''
    def addActivationLayer(self, ActivationClassType):
        layer = {
            'data' : [ ActivationClassType() for i in range(self._last_input_dim)],
            'input_dim' : self._last_input_dim,
            'output_dim' : self._last_input_dim,
            'type' : 'activation'
        }
        self._network.append(layer)
        pass

    '''
    Set the loss function for the optimization
    '''
    def setLoss(self, LossClass):
        self._loss = LossClass(self._output_dim)

    '''
    Process input_data by the network
    '''
    def _forward(self, input_data):
        layer_input = input_data
        layer_i = 0
        for layer in self._network:
            # temporary layer output
            layer_output = np.zeros(layer['output_dim'])

            if layer['type'] == 'neuron':
                # feed each block of the layer with the input data
                b_i = 0
                for b in layer['data']:
                    layer_output[b_i] = b.forward(layer_input)
                    b_i += 1
            elif layer['type'] == 'activation':
                # one to one connection between input data and output data elements
                b_i = 0
                for b in layer['data']:
                    layer_output[b_i] = b.forward(layer_input[b_i])
                    b_i += 1
            else:
                raise ValueError('unknow layer type %s, can not run the network' % layer['type'])

            # current output is the input of the next layer
            layer_input = layer_output
            layer_i += 1

        return layer_output

    '''
    Back propagate through the entire network
    '''
    def _backward(self, output_grad):
        layer_output_grad = output_grad
        layer_i = 0
        for layer in reversed(self._network):
            # temporary layer input gradient
            layer_input_grad = np.zeros(layer['input_dim'])

            if layer['type'] == 'neuron':
                b_i = 0
                for b in layer['data']:
                    layer_input_grad += b.backward(layer_output_grad[b_i])
                    b_i += 1
            elif layer['type'] == 'activation':
                b_i = 0
                for b in layer['data']:
                    layer_input_grad[b_i] = b.backward(layer_output_grad[b_i])
                    b_i += 1
            else:
                raise ValueError('unknow layer type %s, can not run the network' % layer['type'])

            layer_output_grad = layer_input_grad
            layer_i += 1
        return

    '''
    Trigger a weight update for every block of the network
    '''
    def _update(self, learn_rate):
        for layer in self._network:
            for b in layer['data']:
                b.update(learn_rate)

    '''
    Train the network with examples
    (X,Y) training data set (each row is an example)
    '''
    def train(self, X, Y, epoch, learn_rate):
        num_example = X.shape[1]
        loss_historic = np.zeros(epoch)
        for e in range(epoch):
            l_mean = 0.0
            for x, y in zip(X, Y):
                # do forward pass
                network_output = self._forward(x)
                # compute loss
                self._loss.setExpectedData(y)
                l_mean += self._loss.forward(network_output)
                # back propagate
                l_grad = self._loss.backward()
                self._backward(l_grad)
                # update weight
                self._update(learn_rate)
            loss_historic[e] = l_mean / num_example
            print("epoch %d/%d average loss %f" % (e, epoch, loss_historic[e]))
        return loss_historic

    '''
    Process a bunch of input data
    '''
    def run(self, input_data):
        input_data = input_data.reshape((-1, self._input_dim))
        y = np.zeros((input_data.shape[0], self._output_dim))
        i = 0
        for x in input_data:
            y[i, :] = self._forward(x)
            i += 1
        return y

def test_blocks():
    # loss forward backward test
    loss_expect_x = np.random.rand(3)
    loss_x = np.random.rand(3)
    loss = LossL2(3)
    loss.setExpectedData(loss_expect_x)
    loss_y = loss.forward(loss_x)

    loss_grad_x = loss.backward()
    loss_num_grad_x = np.zeros(3)
    dx = 0.000001

    for i in range(3):
        h_loss_x = np.array(loss_x)
        h_loss_x[i] += dx
        h_loss_y = loss.forward(h_loss_x)
        loss_num_grad_x[i] = (h_loss_y - loss_y) / dx

    print("loss forward backward test")
    print("loss expected input :     " + str(loss_expect_x))
    print("loss input :              " + str(loss_x))
    print("loss output :              " + str(loss_y))
    print("loss gradient :           " + str(loss_grad_x))
    print("loss numerical gradient : " + str(loss_num_grad_x))
    print("")

    # relu forward backward test
    relu = Relu()
    relu_x = -0.5
    relu_y = relu.forward(relu_x)
    relu_grad_x = relu.backward(1)
    h_relu_x = relu_x + dx
    h_relu_y = relu.forward(h_relu_x)
    relu_num_grad_x = (h_relu_y - relu_y) / dx

    print("relu forward backward test")
    print("relu input :              " + str(relu_x))
    print("relu output :              " + str(relu_y))
    print("relu gradient :           " + str(relu_grad_x))
    print("relu numerical gradient : " + str(relu_num_grad_x))
    print("")

    # neuron forward backward test
    neuron = Neuron(3)
    neuron_x = np.random.rand(3)
    neuron_weight = np.array(neuron.weight)
    neuron_bias = neuron.bias
    neuron_y = neuron.forward(neuron_x)
    neuron_grad_x = neuron.backward(1)
    neuron_grad_weight = np.array(neuron.grad_weight)
    neuron_grad_bias = np.array(neuron.grad_bias)

    neuron_num_grad_x = np.zeros(3)
    for i in range(3):
        h_neuron_x = np.array(neuron_x)
        h_neuron_x[i] += dx
        h_neuron_y = neuron.forward(h_neuron_x)
        neuron_num_grad_x[i] = (h_neuron_y - neuron_y) / dx

    neuron_num_grad_weight = np.zeros(3)
    for i in range(3):
        h_neuron_weight = np.array(neuron_weight)
        h_neuron_weight[i] += dx
        neuron.weight = h_neuron_weight
        h_neuron_y = neuron.forward(neuron_x)
        neuron_num_grad_weight[i] = (h_neuron_y - neuron_y) / dx
        # restore weight
        neuron.weight = np.array(neuron_weight)

    h_neuron_bias = neuron_bias + dx
    neuron.bias = h_neuron_bias
    h_neuron_y = neuron.forward(neuron_x)
    neuron_num_grad_bias = (h_neuron_y - neuron_y) / dx
    # restore bias
    neuron.bias = neuron_bias

    print("neuron forward backward test")
    print("neuron weight, bias :              " + str(neuron_weight) + ", " + str(neuron_bias))
    print("neuron input :                     " + str(neuron_x))
    print("neuron output :                    " + str(neuron_y))
    print("neuron gradient x :                " + str(neuron_grad_x))
    print("neuron numerical gradient x :      " + str(neuron_num_grad_x))
    print("neuron gradient weight :           " + str(neuron_grad_weight))
    print("neuron numerical gradient weight : " + str(neuron_num_grad_weight))
    print("neuron gradient bias :             " + str(neuron_grad_bias))
    print("neuron numerical gradient bias :   " + str(neuron_num_grad_bias))
    print("")

def test_network():
    # test backward forward on a network
    model = Network(2, 1)
    model.addNeuronlayer(Neuron, 20)
    model.addActivationLayer(Relu)
    model.addNeuronlayer(Neuron, 20)
    model.addActivationLayer(Relu)
    model.addNeuronlayer(Neuron, 1)


    x = np.random.rand(2)
    y = model.run(x)

    # test gradient on first weight of first neuron
    model._backward([1])
    grad_n0_w0 = model._network[0]["data"][0].grad_weight[0]
    dx = 0.0001
    model._network[0]["data"][0].weight[0] += dx
    h_y = model.run(x)
    num_grad_n0_w0 = (h_y - y) / dx
    print("gradient for first neuron weight / first layer :     " + str(grad_n0_w0))
    print("num gradient for first neuron weight / first layer : " + str(num_grad_n0_w0[0, 0]))

def main():
    #---------------
    # Test functions
    #---------------
    test_blocks()
    test_network()

    #---------------
    # Hyperparameters
    #---------------
    NUM_HIDDEN_NODES = 10
    NUM_HIDDEN_LAYER = 2
    NUM_EXAMPLES = 1000
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.0001

    #---------------
    # Data
    #---------------

    # the objective function the network has to mimic
    function_to_learn = lambda x0, x1: np.add(np.multiply(x0, x1), 2 * x0) + 1

    x_range = 10
    # generate random input data
    X_train = np.random.uniform(-x_range, x_range, (NUM_EXAMPLES, 2))
    # generate matching output data
    Y_train = function_to_learn(X_train[:, 0], X_train[:, 1])

    #---------------
    # Build network
    #---------------
    model = Network(2, 1)
    for h in range(NUM_HIDDEN_LAYER):
        model.addNeuronlayer(Neuron, NUM_HIDDEN_NODES)
        model.addActivationLayer(Relu)
    model.addNeuronlayer(Neuron, 1)

    #---------------
    # Build network
    #---------------
    model.setLoss(LossL2)
    loss_historic = model.train(X_train, Y_train, NUM_EPOCHS, LEARNING_RATE)

    #---------------
    # Plot results
    #---------------

    # plot loss
    plt.figure(1)
    plt.plot(loss_historic)
    plt.xlabel('epoch')
    plt.ylabel('mean loss')
    plt.suptitle('Loss historic', fontsize=20)

    # build regular input mesh
    r = np.arange(-x_range, x_range, 1.0)
    plot_mesh_x0, plot_mesh_x1 = np.meshgrid(r, r)
    plotx = np.hstack((plot_mesh_x0.reshape((-1, 1)), plot_mesh_x1.reshape((-1, 1))))
    # run objective function on mesh
    plot_mesh_validy = function_to_learn(plotx[:, 0], plotx[:, 1]).reshape((-1, 1)).reshape(plot_mesh_x0.shape)
    # run model on mesh
    plot_mesh_y = model.run(plotx).reshape(plot_mesh_x0.shape)

    # plot objective function vs model
    fig = plt.figure(2)
    ax = fig.gca(projection='3d')
    ax.plot_surface(plot_mesh_x0, plot_mesh_x1, plot_mesh_validy, alpha=0.5, rstride=1, cstride=1, cmap=cm.autumn, linewidth=0.5, antialiased=False)
    ax.plot_surface(plot_mesh_x0, plot_mesh_x1, plot_mesh_y, alpha=0.5, rstride=1, cstride=1, cmap=cm.winter, linewidth=0.5, antialiased=False)
    ax.set_xlabel('X0 input')
    ax.set_ylabel('X1 input')
    ax.set_zlabel('Y output')
    fig.suptitle('Objective function Vs Neural network', fontsize=20)
    plt.show()

if __name__ == "__main__":
    main()

