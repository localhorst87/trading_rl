from abc import ABC, abstractmethod
from Environment import *
import math
import tensorflow as tf
import pickle
import random

class Agent(ABC):

    def __init__(self):
        super().__init__()
        self.session = tf.Session()
        self.episode = 0

    def __del__(self):
        self.session.close()

    def _episodeWrapper(func):
        ''' decorator for runEpisode. While in the runEpisode method should be defined the arbitrary stuff as handling of the states, actions, rewards, etc.
            the _episodeWrapper method performs the procedure around that, as incrementing the step, performing the training etc.
        '''

        def decoratorWrapper(self):

            if self.episode == 0: tf.global_variables_initializer().run(session = self.session)
            func(self) # here the episode itself is performed
            self.episode += 1
            if self.episode % self.stepSizeTraining == 0 and self.episode >= self.episodeStartTraining:
                batch = self._createBatch()
                self.networkHandler.train(batch, self.learningRate)

        return decoratorWrapper

    @abstractmethod
    def _createNetworks(self):
        pass

    @abstractmethod
    def getAction(self):
        pass

    @abstractmethod
    def runEpisode(self):
        pass

    @abstractmethod
    def _createBatch(self):
        pass

    @property
    def networkHandler(self):
        pass

    @property
    def environment(self):
        pass

    @property
    def learningRate(self):
        pass

    @property
    def episodesStartTraining(self):
        pass

    @property
    def stepSizeTraining(self):
        pass

    @property
    def batchSize(self):
        pass


class OpeningAgent(Agent):

    def __init__(self, environment):
        super().__init__()
        self._environment = environment
        self._learningRate = 1e-4
        self._stepSizeTraining = 50
        self._episodeStartTraining = 100
        self._batchSize = 32
        self.minRewardTarget = 0.33
        self.probabilityThreshold = 0.80
        self.randomActionProbability = [0.15, 0.00] # [start, final]
        self.episodesRandomAction = [100, 300] # [start descent, episode reaching final probability]
        self.possibleActions = self.environment.actionSpace.possibleActions["observe"]
        self.observeAction = self.environment.actionSpace.possibleActions["observe"][0]
        self.openingActions = self.environment.actionSpace.possibleActions["observe"][1:3]
        self.nOpeningActions = len(self.openingActions)
        self.nInputs = 3
        self.replayMemory = ReplayMemory(1000)
        self._createNetworks()

    def _createNetworks(self):
        ''' creates a networkHandler according to the following configuration '''

        tf.reset_default_graph()
        input = tf.placeholder(tf.float32, [None, self.nInputs, self.environment.windowLength])
        network = LstmNormalDistribution("observeNet", input, self.nOpeningActions)
        self.networkHandler = NormalDistributionHandler(network, self.session)

    @Agent._episodeWrapper
    def runEpisode(self):
        ''' executes exactly one single episode '''

        observation = self.environment.reset()
        state = self._createInput(observation)
        tradingState = self.environment.getTradingState()

        while tradingState == "observe":

            action = self._getAction(state) # action string, e.g. "keep_observing"
            nextObservation, reward, tradingState, _ = self.environment.step(action)
            nextState = self._createInput(nextObservation)

            if action != self.observeAction:
                sample = self._createSample(state, action, reward)
                self.replayMemory.add(sample)

            state = nextState

    def _getAction(self, state):
        ''' returns the action to perform according to an epsilon greedy strategy as the agent
            works with deterministic actions

            IN      state       (list)      state input vectors for the network
            OUT     action      (string)    action expression of the chosen action
        '''

        if self._getRandomActionProb() > random.uniform(0, 1):
            action = self._getRandomAction()
        else:
            action = self._getBestAction(state)

        return action

    def _getRandomActionProb(self):
        ''' calculates the random probability for actions according to the start and final values of
            probabilities and episodes

            OUT     (float)     random action probability
        '''

        if self.episode <= self.episodesRandomAction[0]:
            actionProb = self.randomActionProbability[0]
        else:
            gradient = (self.randomActionProbability[1] - self.randomActionProbability[0]) / (self.episodesRandomAction[1] - self.episodesRandomAction[0])
            actionProb = self.randomActionProbability[0] + (self.episode - self.episodesRandomAction[0]) * gradient

        return max(0, actionProb)

    def _getBestAction(self, state):
        ''' returns the best action according to the prediction of the network

            IN      state       (list)      state input vectors for the network
            OUT     action      (string)    action expression of the best action
        '''

        meanEstimation, varianceEstimation = self._getEstimation(state)
        winProbLong = self._calcWinProb(meanEstimation[0], varianceEstimation[0])
        winProbShort = self._calcWinProb(meanEstimation[1], varianceEstimation[1])
        winProbabilities = [winProbLong, winProbShort]

        bestAction = np.argmax(winProbabilities)

        if winProbabilities[bestAction] >= self.probabilityThreshold:
            action = self.openingActions[bestAction]
        else:
            action = self.observeAction

        return action

    def _getRandomAction(self):
        ''' returns a random action from the possible actions

            OUT     (string)    action string of the random action
        '''

        randomActionNum = random.randint( 0, len(self.possibleActions) - 1 )

        return self.possibleActions[randomActionNum]

    def _calcWinProb(self, mean, variance):
        ''' calculates the probability to get a reward of more than the self.minRewardTarget

            IN      mean            (float)     predicted mean reward of the network
            IN      variance        (float)     predicted variance of the network
            OUT     winProbability  (float)     probability to gain more than the minRewardTarget
        '''

        cumulativeDistribution = 0.5 * (1 + math.erf( (self.minRewardTarget - mean)/math.sqrt(2*variance) )) # estimates the relative amount of points with a reward < minRewardTarget
        winProbability = 1 - cumulativeDistribution

        return winProbability

    def _getEstimation(self, state):
        ''' returns the estimation of mean and risk of the current state for both opening actions (long/short)

            IN      state               (list)      state vectors from environment
            OUT     meanEstimation      (list)      estimated mean values for long and short action
            OUT     varianceEstimation  (list)      estimated variance for long and short action
        '''

        networkPrediction = self.networkHandler.predict(self.session, state)
        meanEstimation = networkPrediction[0]
        varianceEstimation = networkPrediction[1]

        return meanEstimation, varianceEstimation

    def _createSample(self, state, action, reward):
        ''' creates a Sample of the last observation

            IN      state   (list)      state vectors before action returned by the environment
            IN      action  (string)    action expression of the performed action
            IN      reward  (float)     last reward returned by the environment
            OUT     sample  (Sample)    standardized sample of one step
        '''

        actionNumber = self.possibleActions.index(action)

        sample = Sample()
        sample.state = state
        sample.reward = reward
        sample.action =  actionNumber

        return sample

    def _createInput(self, observation):
        ''' creates an network input ("state") of the observation from the environment

            IN      observation     (Pandas.Dataframe)      observation of the environment
            OUT                     (list)                  states that can be used as an input for the LSTM network
        '''

        macdHistogramNormed = observation["macd_histogram"].values / observation["stddev"].values
        adx = observation["adx"].values
        rsi = observation["rsi"].values

        return [macdHistogramNormed.tolist(), adx.tolist(), rsi.tolist()]

    def _createBatch(self):
        ''' creates a MonteCarlo batch from the replay memory

            OUT     (Batch)     batch of the size specified in the properties
        '''

        batch = Batch(self.batchSize)
        randomSamples = self.replayMemory.getSamples(self.batchSize)

        for sample in randomSamples:
            batch.add(sample) if not batch.isFull() else break

        return batch

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, value):
        self._environment = value

    @property
    def learningRate(self):
        return self._learningRate

    @learningRate.setter
    def learningRate(self, value):
        self._learningRate = value

    @property
    def episodesStartTraining(self):
        return self._episodesStartTraining

    @episodesStartTraining.setter
    def episodesStartTraining(self, value):
        self._episodesStartTraining = value

    @property
    def stepSizeTraining(self):
        return self._environment

    @stepSizeTraining.setter
    def stepSizeTraining(self, value):
        self._stepSizeTraining = value

    @property
    def batchSize(self):
        return self._batchSize

    @batchSize.setter
    def batchSize(self, value):
        self._batchSize = value

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

        IN      session     (tf.Session)    tensorflow session
        IN      input       (list)          input that fulfills the input placeholder
        OUT                 (list)          prediction of the neural network '''

        return session.run(self.network.output, feed_dict = {self.network.input:input} )

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.session, path)

    def load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.session, path)

class NormalDistributionHandler(NetworkHandler):
    '''
    Training of a network that outputs a normal distribution (mean value and variance of actions)

    INPUTS:
        network           neural network to train
    '''

    def __init__(self, network, session):
        super().__init__()
        self._session = session
        self._network = network
        self._loss = None
        self._training = None
        self.learningRate = tf.placeholder(tf.float32)
        self.actionPerformed = tf.placeholder(tf.int32, shape = (None,) )
        self.reward = tf.placeholder(tf.float32, shape = (None,) )
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
    def training(self):
        return self._training

    @training.setter
    def training(self):
        self._training = value

    def _defineOptimization(self):
        ''' defines the graph for training the network'''

        mean = self.network.output[0] * tf.one_hot(self.actionPerformed, self.network.nOutputs)
        variance = self.network.output[1] * tf.one_hot(self.actionPerformed, self.network.nOutputs)
        self.loss = tf.reduce_sum( 0.5 * ( tf.log(2 * math.pi * variance) + tf.square(self.reward - mean) / variance ) ) # natural log of normal distribution

        optimizer = tf.train.AdamOptimizer(learning_rate = self.learningRate)
        self.training = optimizer.minimize(self.loss)

    def train(self, batch, learningRate):
        ''' performs the training of the network according to the defined optimization

            IN      batch           (Batch)     batch of state, action, reward and next state samples
            IN      learningRate    (float)     learning rate for training the network
            OUT                     (float)     loss of the recent training step '''

        trainingLoss, _ = self.session.run( [self.loss, self.training], feed_dict = {self.network.input:batch.states, self.actionPerformed:batch.actions, self.reward:batch.rewards, self.learningRate:learningRate} )

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
    def output(self):
        pass

    @property
    def nOutputs(self):
        pass

    @abstractmethod
    def _buildNetwork(self):
        pass

class LstmNormalDistribution(Network):
    '''
    Concrete Network class. Implements a LSTM with two heads of fully connected layers
    that intends to implement mean and variance values of actions

    INPUTS:
        networkName     (string)            name of the network
        input           (tf.placeholder)    tensorflow placeholder for network inputs
        nActions        (int)               number of all actions
    '''

    def __init__(self, networkName, input, nActions):
        super().__init__()
        self._networkName = networkName
        self._input = input
        self._output = None
        self.lstmUnits = 64
        self.neuronsFc = 16
        self._nOutputs = nActions
        self._buildNetwork()

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

    def _buildNetwork(self):
        ''' creates the graph of the neural network:

              --> FCL --> FCL (mean values of actions)
        LSTM
              --> FCL --> FCL (variance of actions)
        '''
        with tf.variable_scope(self.networkName) as scope:

            # LSTM
            lstmCell = tf.contrib.rnn.LSTMCell(self.lstmUnits)
            wrappedLstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob = 0.9)
            lstmOutputs, _ = tf.nn.dynamic_rnn(wrappedLstmCell, self.input, dtype = tf.float32)  # shape: [batchSize, sequenceLength, lstmUnits]
            lstmOutputs = tf.transpose(lstmOutputs, [1, 0, 2]) # shape: [sequenceLength, batchSize, lstmUnits]
            sequenceLength = int( lstmOutputs.get_shape()[0] )
            lastSequenceSample = tf.gather( lstmOutputs, sequenceLength - 1 ) # returns tensorf of shape [batchSize, lstmUnits] of the last sequence sample

            # divided fully connected Layers
            weightsInitFcMean = tf.truncated_normal_initializer( stddev = math.sqrt( 2 / (self.lstmUnits + self.neuronsFc) ) )
            biasInitFcMean = tf.truncated_normal_initializer( stddev = math.sqrt(2 / self.neuronsFc) )
            fcMeanLayer = tf.contrib.layers.fully_connected( lastSequenceSample, num_outputs = self.neuronsFcn1, weights_initializer = weightsInitFc, biases_initializer = biasInitFc)

            weightsInitFcVariance = tf.truncated_normal_initializer( stddev = math.sqrt( 2 / (self.lstmUnits + self.neuronsFc) ) )
            biasInitFcVariance = tf.truncated_normal_initializer( stddev = math.sqrt(2 / self.neuronsFc) )
            fcVarianceLayer = tf.contrib.layers.fully_connected( lastSequenceSample, num_outputs = self.neuronsFcn1, weights_initializer = weightsInitFc, biases_initializer = biasInitFc)

            # output heads for mean and variance
            weightsInitOutputMean = tf.truncated_normal_initializer( stddev = math.sqrt( 2 / (self.neuronsFc + self.nOutputs) ) )
            biasInitOutputMean = tf.constant_initializer(0)
            outputMean = tf.contrib.layers.fully_connected( fcMeanLayer, num_outputs = self.nOutputs, activation_fn = None, weights_initializer = weightsInitOutputMean, biases_initializer = biasInitOutputMean)

            weightsInitOutputVariance = tf.truncated_normal_initializer( stddev = math.sqrt( 2 / (self.neuronsFc + self.nOutputs) ) )
            biasInitOutputVariance = tf.constant_initializer(0)
            outputVariance = tf.contrib.layers.fully_connected( fcVarianceLayer, num_outputs = self.nOutputs, activation_fn = tf.sigmoid, weights_initializer = weightsInitOutputVariance, biases_initializer = biasInitOutputVariance)

            # wrapped outputs
            self.output = [outputMean, outputVariance]

class Batch:
    '''
    Batch of states, actions, rewards and next states.

    INPUTS:
        size    (int)       size of the batch
    '''

    def __init__(self, size):
        self.size = size
        self.states = []
        self.actions = []
        self.rewards = []
        self.nextStates = []
        self.dones = []

    def add(self, sample):
        ''' adds a new sample to the batch if the number of samples in the batch is smaller than the batch size

        IN      sample      (dict)      dictionary with the fields state, action, reward, nextState and done '''

        if not sample.isSet: raise ValueError("sample is missing data")

        if not self.isFull():
            self.states.append(sample.state)
            self.actions.append(sample.action)
            self.rewards.append(sample.reward)
            self.nextStates.append(sample.nextState)
            self.dones.append(sample.done)

    def isFull(self):
        ''' checks if the batch is full

        OUT         (bool)      returns True if the number of entries is equal to the defined batch size '''

        if len(self.actions) >= self.size:
            return True
        else:
            return False
