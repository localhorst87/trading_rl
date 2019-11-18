from abc import ABC, abstractmethod
import math
import tensorflow as tf
import pickle

class NetworkHandler(ABC):
    '''
    The NetworkHandler is an abstract class for the handling of neural networks.

    It defines the optimization and performs the training of a network.
    '''

    def __init__(self):
        super().__init__()

    @property
    def session(self):
        pass

    @property
    def graph(self):
        pass

    @property
    def network(self):
        pass

    @property
    def loss(self):
        pass

    @property
    def training(self):
        pass

    @abstractmethod
    def train(self, batch, learningRate):
        pass

    def predict(self, session, input):
        ''' returns the prediction of the neural network with the given input

        IN      session         (tf.Session)    tensorflow session
        IN      input           (list)          input that fulfills the input placeholder
        OUT     prediction      (list)          prediction of the neural network '''

        with self.graph.as_default():

            prediction = session.run(self.network.output, feed_dict = {self.network.input:[input]} )

        return prediction

    def save(self, path):
        ''' saves the graph of the network to the given path '''

        with self.graph.as_default():

            saver = tf.train.Saver()
            saver.save(self.session, path)

    def load(self, path):
        ''' reloads a saved graph of the network from the given path '''

        with self.graph.as_default():

            saver = tf.train.Saver()
            saver.restore(self.session, path)

class NormalDistributionHandler(NetworkHandler):
    '''
    Training of a network that outputs a normal distribution (mean value and variance of actions)

    INPUTS:
        network           neural network to train
    '''

    def __init__(self, network, session, graph):
        super().__init__()
        self._session = session
        self._graph = graph
        self._network = network
        self._loss = None
        self._training = None
        with self.graph.as_default():
            self.learningRate = tf.placeholder(tf.float32)
            self.actionPerformed = tf.placeholder(tf.int32, shape = (None,) )
            self.reward = tf.placeholder(tf.float32, shape = (None,) )
            self.meanValues = tf.placeholder(tf.float32, shape = (None, ) )
            self._network.buildNetwork()
            self._defineOptimization()

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, value):
        self._session = value

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        pass

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        self._loss = value

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, value):
        pass

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, value):
        self._training = value

    def _defineOptimization(self):
        ''' defines the graph for training the network'''

        mean = tf.reduce_sum(self.network.output[0] * tf.one_hot(self.actionPerformed, self.network.nOutputs), axis = -1)
        variance = tf.reduce_sum(self.network.output[1] * tf.one_hot(self.actionPerformed, self.network.nOutputs), axis = -1)

        self.loss = tf.reduce_mean( tf.square(self.reward - mean) / variance + 0.5 * ( tf.log(2 * math.pi * variance)) ) # natural log of normal distribution
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate)
        self.training = optimizer.minimize(self.loss)

    def train(self, batch, learningRate, **additionalParameters):
        ''' performs the training of the network according to the defined optimization

            IN      batch                   (Batch)     batch of state, action, reward and next state data
            IN      learningRate            (float)     learning rate for training the network
            IN      **additionalParameters  (dict)      additional parameters as key word arguments. no use in this function, but required to fulfill generic function call
            OUT                             (float)     loss of the recent training step '''

        with self.graph.as_default():

            trainingLoss, _ = self.session.run( [self.loss, self.training], feed_dict = {self.network.input:batch.states, self.actionPerformed:batch.actions, self.reward:batch.rewards,    self.learningRate:learningRate} )

        return trainingLoss

class A2CHandler(NetworkHandler):
    '''

    '''

    def __init__(self, network, session, graph):
        super().__init__()
        self._session = session
        self._graph = graph
        self._network = network
        self._loss = None
        self._training = None
        self._nOpenTypes = 2 # number of different opening types
        with self.graph.as_default():
            self.learningRate = tf.placeholder(tf.float32)
            self.qValue = tf.placeholder(tf.float32, shape = (None, ) )
            self.openType = tf.placeholder(tf.int32) # open long (0) or open short position (1)
            self.actionPerformed = tf.placeholder(tf.int32, shape = (None, ) )
            self._network.buildNetwork()
            self._defineOptimization()

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, value):
        self._session = value

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        pass

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        self._loss = value

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, value):
        pass

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, value):
        self._training = value

    def _defineOptimization(self):
        ''' defines the graph for training the network'''

        actorActionProbs = tf.gather(self.network.output[0], self.openType) # contains the estimation for an open long AND open short position
        criticValues = tf.gather(self.network.output[1], self.openType) # contains the estimation for an open long AND open short position

        actorProb = tf.reduce_sum(actorActionProbs * tf.one_hot(self.actionPerformed, self.network.nOutputs), axis = -1) # right value according to performed action
        clippedActorProb = tf.clip_by_value(actorProb, 1e-4, 1)
        value = tf.reduce_sum(criticValues, axis = -1) # reduces dimension
        advantage = self.qValue - value

        lossActor = tf.reduce_mean( -tf.log(clippedActorProb) * advantage )
        lossCritic = tf.reduce_mean( tf.square(advantage) )
        self.loss = lossActor + lossCritic
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate)
        self.training = optimizer.minimize(self.loss)

    def train(self, batch, learningRate, **additionalParameters):
        ''' performs the training of the network according to the defined optimization

            IN      batch                   (Batch)     batch of state, action, reward and next state data
            IN      learningRate            (float)     learning rate for training the network
            IN      **additionalParameters  (dict)      additional parameters as key word arguments. Here used to indicate the type of the open position (long=0, short=1)
            OUT                             (float)     loss of the recent training step '''

        tradeType = additionalParameters["tradeType"]

        with self.graph.as_default():

            trainingLoss, _ = self.session.run( [self.loss, self.training], feed_dict = {self.openType:tradeType, self.network.input:batch.states, self.actionPerformed:batch.actions, self.qValue:batch.rewards, self.learningRate:learningRate} )

        return trainingLoss

class Network(ABC):
    '''
    Interface for any kind of neural network

    Inside the concrete classes the graph of the neural network is created
    '''

    def __init__(self):
        super().__init__()

    @property
    def networkName(self):
        pass

    @property
    def input(self):
        pass

    @property
    def inputShape(self):
        pass

    @property
    def output(self):
        pass

    @property
    def nOutputs(self):
        pass

    @abstractmethod
    def buildNetwork(self, graph):
        pass

class CnnNormalDistribution(Network):
    '''
    Concrete Network class. Implements a convolutional neural network with two heads
    of fully connected layers that intends to implement mean and variance values of actions

    INPUTS:
        networkName     (string)            name of the network
        inputShape      (list)              shape of the network input
        nActions        (int)               number of all actions
    '''

    def __init__(self, networkName, inputShape, nActions):
        super().__init__()
        self._networkName = networkName
        self._inputShape = inputShape
        self._input = None
        self._output = None
        self.nFeatures = int(inputShape[1])
        self._nOutputs = nActions

    @property
    def networkName(self):
        return self._networkName

    @networkName.setter
    def networkName(self, value):
        pass

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value

    @property
    def inputShape(self):
        return self._inputShape

    @inputShape.setter
    def inputShape(self, value):
        pass

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value

    @property
    def nOutputs(self):
        return self._nOutputs

    @nOutputs.setter
    def nOutputs(self, value):
        pass

    def buildNetwork(self):

        self.input = tf.placeholder(tf.float32, shape = self.inputShape)

        # first smooth the data
        pool0 = tf.layers.average_pooling2d(inputs = self.input, pool_size = [1, 3], strides = (1, 1), padding = "same", name = "OA_pool0")

        # mean estimation convolutional layers
        convKernelInit = tf.truncated_normal_initializer(stddev = 0.2)
        conv1Mean = tf.layers.conv2d(inputs = pool0, filters = 16, kernel_size = [self.nFeatures, 1], strides = (1, 1), kernel_initializer = convKernelInit, name = "OA_conv1Mean")
        pool1Mean = tf.layers.average_pooling2d(inputs = conv1Mean, pool_size = [1, 3], strides = (1, 1), padding = "same", name = "OA_pool1Mean")
        conv2Mean = tf.layers.conv2d(inputs = pool1Mean, filters = 4, kernel_size = [1, 3], strides = (1, 1), padding = "same", kernel_initializer = convKernelInit, name = "OA_conv2Mean")
        pool2Mean = tf.layers.average_pooling2d(inputs = conv2Mean, pool_size = [1, 3], strides = (1, 1), padding = "same", name = "OA_pool2Mean")

        # configure flattened fully connected layer for mean estimation
        nDatapointsMean = int( pool2Mean.get_shape()[2] )
        nLayersMean = int( pool2Mean.get_shape()[3] )
        nNeuronsMean = int( nDatapointsMean * nLayersMean )
        weightsInitFcMean = tf.truncated_normal_initializer(stddev = math.sqrt( 2/(nNeuronsMean + 16) ) )
        weightsInitOutputMean = tf.truncated_normal_initializer(stddev = 0.4)

        # mean estimation fully connected layers and output
        flattenedLayerMean = tf.reshape(pool2Mean, [-1, nNeuronsMean], name = "OA_flattenedLayerMean")
        dropoutLayerMean = tf.layers.dropout(flattenedLayerMean, rate = 0.20, name = "OA_dropoutLayerMean")
        fullyConnectedMean = tf.layers.dense(dropoutLayerMean, units = 16, activation = tf.nn.leaky_relu, kernel_initializer = weightsInitFcMean, name = "OA_fullyConnectedMean")
        outputMean = tf.layers.dense(fullyConnectedMean, units = self.nOutputs, activation = None, kernel_initializer = weightsInitOutputMean, name = "OA_outputMean")

        # convolutional layers for the variance estimation
        conv1Variance = tf.layers.conv2d(inputs = pool0, filters = 16, kernel_size = [self.nFeatures, 1], strides = (1, 1), kernel_initializer = convKernelInit, name = "OA_conv1Variance")
        pool1Variance = tf.layers.average_pooling2d(inputs = conv1Variance, pool_size = [1, 3], strides = (1, 1), name = "OA_pool1Variance")
        conv2Variance = tf.layers.conv2d(inputs = pool1Variance, filters = 4, kernel_size = [1, 3], strides = (1, 1), padding = "same", kernel_initializer = convKernelInit, name = "OA_conv2Variance")
        pool2Variance = tf.layers.average_pooling2d(inputs = conv2Variance, pool_size = [1, 3], strides = (1, 1), padding = "same", name = "OA_pool2Variance")

        # configure flattened fully connected layer for variance estimation
        nDatapointsVariance = int( pool2Variance.get_shape()[2] )
        nLayersVariance = int( pool2Variance.get_shape()[3] )
        nNeuronsVariance = int( nDatapointsVariance * nLayersVariance )
        weightsInitFcVariance = tf.truncated_normal_initializer(stddev = math.sqrt( 2/(nNeuronsVariance + 16) ) )
        weightsInitOutputVariance = tf.truncated_normal_initializer(stddev = 0.4)

        # variance estimation fully connected layers and output
        flattenedLayerVariance = tf.reshape(pool2Variance, [-1, nNeuronsVariance], name = "OA_flattenedLayerVariance")
        dropoutLayerVariance = tf.layers.dropout(flattenedLayerVariance, rate = 0.15, name = "OA_dropoutLayerVariance")
        fullyConnectedVariance = tf.layers.dense(dropoutLayerVariance, units = 16, activation = tf.nn.leaky_relu, kernel_initializer = weightsInitFcVariance, name = "OA_fullyConnectedVariance")
        outputVariance = tf.layers.dense(fullyConnectedVariance, units = self.nOutputs, activation = tf.math.softplus, kernel_initializer = weightsInitOutputVariance, name = "OA_outputVariance")

        # wrapped outputs
        self.output = [outputMean, outputVariance]

class LstmA2C(Network):
    '''
    Concrete Network class. Implements two independent LSTMs (opened long and opened short position) with two heads (actor and critic)

    INPUTS:
        networkName     (string)            name of the network
        inputShape      (list)              shape of the network input
        nActions        (int)               number of all actions
    '''

    def __init__(self, networkName, inputShape, nActions):
        super().__init__()
        self._networkName = networkName
        self._inputShape = inputShape
        self._output = None
        self._input = None
        self._nOutputs = nActions
        self._hiddenSize = 64

    @property
    def networkName(self):
        return self._networkName

    @networkName.setter
    def networkName(self, value):
        pass

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value

    @property
    def inputShape(self):
        return self._inputShape

    @inputShape.setter
    def inputShape(self, value):
        pass

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value

    @property
    def nOutputs(self):
        return self._nOutputs

    @nOutputs.setter
    def nOutputs(self, value):
        pass

    @staticmethod
    def _getSequenceLength(transposedInput):
        ''' return the sequence length of the batch on the premise that all non used values are filled with Nones / NaNs.
            transposed input must be of size (batch size, sequence length, number of features)

            IN      transposedInput     (tf.Tensor)     input sequence transposed to the above mentioned size
            OUT     length              (array)         length of the sequence for each state in the batch
        '''

        boolMaskNoneValues = tf.reduce_any(tf.is_nan(transposedInput), 2)
        usedPositions = tf.cast(tf.logical_not(boolMaskNoneValues), tf.int32)
        length = tf.reduce_sum(usedPositions, 1)

        return length

    def buildNetwork(self):

        self.input = tf.placeholder(tf.float32, shape = self.inputShape) # shape [batch size, number of features, sequence length]
        inputTransposed = tf.transpose(self.input, [0, 2, 1]) # transpose input because we have a standardized input shape in the agent
        sequenceLength = self._getSequenceLength(inputTransposed)
        cleanInput = tf.where(tf.is_nan(inputTransposed), tf.zeros_like(inputTransposed), inputTransposed) # same as self.input but with NaN replaced with 0 to not cause training error

        lstmCellLong = tf.nn.rnn_cell.LSTMCell(self._hiddenSize, name = "CA_lstmCellLong")
        lstmCellShort = tf.nn.rnn_cell.LSTMCell(self._hiddenSize, name = "CA_lstmCellShort")

        outputLong, _ = tf.nn.dynamic_rnn(cell = lstmCellLong, inputs = cleanInput, sequence_length = sequenceLength, dtype = tf.float32) # shape [batch size, sequence length, LSTM units]
        outputShort, _ = tf.nn.dynamic_rnn(cell = lstmCellShort, inputs = cleanInput, sequence_length = sequenceLength, dtype = tf.float32)

        outputIndices = tf.expand_dims(sequenceLength-1, axis = -1)
        lastOutputLong = tf.batch_gather(outputLong, outputIndices) # we are interested only in the last output, shape [1, batch size, LSTM Units]
        lastOutputShort = tf.batch_gather(outputShort, outputIndices)

        lastOutputActorLong, lastOutputCriticLong = tf.split(lastOutputLong, 2, axis = -1) # split the output to not have negative interaction of critic and actor
        lastOutputActorShort, lastOutputCriticShort = tf.split(lastOutputShort, 2, axis = -1)

        weightsInitOutput = tf.truncated_normal_initializer(stddev = 0.4)

        outputActorLong = tf.layers.dense(lastOutputActorLong, units = self.nOutputs, activation = tf.nn.softmax, kernel_initializer = weightsInitOutput, name = "CA_outputActorLong")
        outputCriticLong = tf.layers.dense(lastOutputCriticLong, units = 1, activation = None, kernel_initializer = weightsInitOutput, name = "CA_outputCriticLong")

        outputActorShort = tf.layers.dense(lastOutputActorShort, units = self.nOutputs, activation = tf.nn.softmax, kernel_initializer = weightsInitOutput, name = "CA_outputActorShort")
        outputCriticShort = tf.layers.dense(lastOutputCriticShort, units = 1, activation = None, kernel_initializer = weightsInitOutput, name = "CA_outputCriticShort")

        # wrapped outputs
        self.output = [ [outputActorLong, outputActorShort], [outputCriticLong, outputCriticShort] ]
