from abc import ABC, abstractmethod
from Environment import *
import math
import tensorflow as tf
import pickle
import random

class Agent(ABC):
    '''
    Abstract Agent base class: Performs the exploration and exploitation of the environment and is
    responsible for the training of the policy.

    Minimum methods to implement:
        runEpisode, _createNetworks, _getAction, _createBatch

    Properties to implement:
        environment, networkHandler, learningRate, episodeStartTraining, stepSizeTraining
    '''

    def __init__(self):
        super().__init__()
        self.session = tf.Session()
        self.episode = 0

    def __del__(self):
        self.session.close()

    def _episodeWrapper(runEpisodeFunc):
        ''' decorator for runEpisode. While in the runEpisode method should be defined the arbitrary stuff as handling of the states, actions, rewards, etc.
            the _episodeWrapper method performs the procedure around that, as incrementing the step, performing the training etc.
        '''

        def decoratorWrapper(self):

            if self.episode == 0: tf.global_variables_initializer().run(session = self.session)

            reward, action = runEpisodeFunc(self) # here the episode itself is performed

            self.episode += 1
            if self.episode % self.stepSizeTraining == 0 and self.episode >= self.episodeStartTraining:
                batch = self._createBatch()
                loss = self.networkHandler.train(batch, self.learningRate)
            else:
                loss = None

            return reward, action, loss

        return decoratorWrapper

    @abstractmethod
    def runEpisode(self):
        pass

    @abstractmethod
    def _createNetworks(self):
        pass

    @abstractmethod
    def _getAction(self):
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
    def episodeStartTraining(self):
        pass

    @property
    def stepSizeTraining(self):
        pass


class OpeningAgent(Agent):
    ''' Concrete Agent for estimating when to open a position. The Opening Agent estimates the expected reward
        and a variance of this mean value for opening a long or short position. Based on this values a probability
        to gain a user defined reward (defined by the property minRewardTarget) is calculated. So the user keeps
        the total control over the risk of opening a position by setting a risk threshold (defined by the property
        probabilityThreshold).

        The OpeningAgent uses sequence data that is forwarded to a LSTM network with fully connected layers.
        It follows a stricty value based and deterministic policy. The training is performed via MonteCarlo
        sampling from a replay memory, as it does not uses policy gradients.

        INPUTS:
            environment             (Environment)   The environment to sample and receive rewards from

        CONFIGURABLE PROPERTIES:
            batchSize               (int)           number of episodes inside a single training batch
            learningRate            (float)         learning Rate for training
            stepSizeTraining        (int)           intervall of episodes each time a training will be executed
            episodeStartTraining    (int)           number of episodes to wait until the first training
            randomActionProbability (float)         [0...1] chance of executing a random action instead of using the network
            minRewardTarget         (float)         [0...1] target reward for creating the success probabilities
            probabilityThreshold    (float)         (0...1) if a success probability for a long or short action exceeds this values,
                                                    a position will be opened
    '''

    def __init__(self, environment):
        super().__init__()
        self.minRewardTarget = 0.00
        self.probabilityThreshold = 0.50
        self.randomActionProbability = 0.00
        self.replayMemory = ReplayMemory(15000)
        self.batchSize = 128
        self._learningRate = 1e-4
        self._stepSizeTraining = 75
        self._episodeStartTraining = 1000
        self._networkLstmUnits = 150
        self._networkSizeFullyConnected1 = 75
        self._networkSizeFullyConnected2 = 25
        self._environment = environment
        self._minIterationsToOpen = 50 # wait this number of iterations until opening a position will be possible
        self._minSamplesLeft = 100 # abort episode after this number of (or less) remaining samples
        self._possibleActions = self.environment.actionSpace.possibleActions["observe"]
        self._observeAction = self.environment.actionSpace.possibleActions["observe"][0]
        self._openingActions = self.environment.actionSpace.possibleActions["observe"][1:3]
        self._nOpeningActions = len(self._openingActions)
        self._nInputs = 1 # number of signals as input
        self._createNetworks()

    def _createNetworks(self):
        ''' creates a networkHandler according to the following configuration '''

        input = tf.placeholder(tf.float32, [None, self._nInputs, self.environment.windowLength])
        network = LstmNormalDistribution("observeNet", input, self._nOpeningActions)
        network.lstmUnits = self._networkLstmUnits
        network.neuronsFc1 = self._networkSizeFullyConnected1
        network.neuronsFc2 = self._networkSizeFullyConnected2
        network.buildNetwork()
        self.networkHandler = NormalDistributionHandler(network, self.session)


    @Agent._episodeWrapper
    def runEpisode(self):
        ''' executes exactly one single episode '''

        try:
            observation = self.environment.reset()
            state = self._createInput(observation)
            tradingState = "observe"

            while tradingState == "observe" and self.environment.dataset.getRemainingSamples() > self._minSamplesLeft:

                action = self._getAction(state) # action string, e.g. "keep_observing"
                nextObservation, reward, tradingState, _ = self.environment.step(action)
                nextState = self._createInput(nextObservation)

                if action != self._observeAction:
                    sample = self._createSample(state, action, reward)
                    self.replayMemory.add(sample)

                state = nextState

        except KeyboardInterrupt:
            raise

        except:
            print("error occured")
            reward = 0
            action = "keep_observing"

        return reward, action

    def _getAction(self, state):
        ''' returns the action to perform according to an epsilon greedy strategy as the agent
            works with deterministic actions

            IN      state       (list)      state input vectors for the network
            OUT     action      (string)    action expression of the chosen action
        '''

        if self.environment.dataset.nIteration < self._minIterationsToOpen:
            action = self._observeAction
        elif self.randomActionProbability > random.uniform(0, 1):
            action = self._getRandomAction()
        else:
            action = self._getBestAction(state)

        return action

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
            action = self._openingActions[bestAction]
        else:
            action = self._observeAction

        return action

    def _getRandomAction(self):
        ''' returns a random action from the possible actions

            OUT     (string)    action string of the random action
        '''

        randomActionNum = random.randint( 0, len(self._possibleActions) - 1 )

        return self._possibleActions[randomActionNum]

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

        return np.squeeze(meanEstimation), np.squeeze(varianceEstimation)

    def _createSample(self, state, action, reward):
        ''' creates a Sample of the last observation

            IN      state   (list)      state vectors before action returned by the environment
            IN      action  (string)    action expression of the performed action
            IN      reward  (float)     last reward returned by the environment
            OUT     sample  (Sample)    standardized sample of one step
        '''

        actionNumber = self._possibleActions.index(action)

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

        priceChange = observation["price_ask_close"].diff().fillna(0).values / observation["price_ask_close"].values

        return [priceChange]

    def _createBatch(self):
        ''' creates a MonteCarlo batch from the replay memory

            OUT     (Batch)     batch of the size specified in the properties
        '''

        batch = Batch(self.batchSize)
        randomSamples = self.replayMemory.getSamples(self.batchSize)

        for sample in randomSamples:
            if not batch.isFull(): batch.add(sample)
            else: break

        return batch

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, value):
        self._environment = value

    @property
    def networkHandler(self):
        return self._networkHandler

    @networkHandler.setter
    def networkHandler(self, value):
        self._networkHandler = value

    @property
    def learningRate(self):
        return self._learningRate

    @learningRate.setter
    def learningRate(self, value):
        self._learningRate = value

    @property
    def episodeStartTraining(self):
        return self._episodeStartTraining

    @episodeStartTraining.setter
    def episodeStartTraining(self, value):
        self._episodeStartTraining = value

    @property
    def stepSizeTraining(self):
        return self._stepSizeTraining

    @stepSizeTraining.setter
    def stepSizeTraining(self, value):
        self._stepSizeTraining = value

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

        return session.run(self.network.output, feed_dict = {self.network.input:[input]} )

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
    def training(self, value):
        self._training = value

    def _defineOptimization(self):
        ''' defines the graph for training the network'''

        mean = tf.reduce_sum(self.network.output[0] * tf.one_hot(self.actionPerformed, self.network.nOutputs) )
        variance = tf.reduce_sum(self.network.output[1] * tf.one_hot(self.actionPerformed, self.network.nOutputs) )
        self.loss = tf.reduce_mean( 0.5 * ( tf.log(2 * math.pi * variance) + tf.square(self.reward - mean) / variance ) ) # natural log of normal distribution

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
    def buildNetwork(self):
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
        self.lstmUnits = 128
        self.neuronsFc1 = 64
        self.neuronsFc2 = 16
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
        ''' creates the graph of the neural network:

              --> FCL --> FCL --> FCL (mean values of actions)
        LSTM
              --> FCL --> FCL --> FCL (variance of actions)
        '''
        with tf.variable_scope(self.networkName) as scope:

            # LSTM
            lstmCell = tf.contrib.rnn.LSTMCell(self.lstmUnits)
            wrappedLstmCell = tf.contrib.rnn.DropoutWrapper(cell = lstmCell, output_keep_prob = 0.92)
            lstmOutputs, _ = tf.nn.dynamic_rnn(wrappedLstmCell, self.input, dtype = tf.float32)  # shape: [batchSize, sequenceLength, lstmUnits]
            lstmOutputs = tf.transpose(lstmOutputs, [1, 0, 2]) # shape: [sequenceLength, batchSize, lstmUnits]
            sequenceLength = int( lstmOutputs.get_shape()[0] )
            lastSequenceSample = tf.gather( lstmOutputs, sequenceLength - 1 ) # returns tensor of shape [batchSize, lstmUnits] of the last sequence sample

            # divided fully connected Layers
            weightsInitFcMean1 = tf.truncated_normal_initializer( stddev = math.sqrt( 2 / (self.lstmUnits + self.neuronsFc1) ) )
            biasInitFcMean1 = tf.truncated_normal_initializer( stddev = math.sqrt(2 / self.neuronsFc1) )
            fcMeanLayer1 = tf.contrib.layers.fully_connected( lastSequenceSample, num_outputs = self.neuronsFc1, weights_initializer = weightsInitFcMean1, biases_initializer = biasInitFcMean1)

            weightsInitFcMean2 = tf.truncated_normal_initializer( stddev = math.sqrt( 2 / (self.neuronsFc1 + self.neuronsFc2) ) )
            biasInitFcMean2 = tf.truncated_normal_initializer( stddev = math.sqrt(2 / self.neuronsFc2) )
            fcMeanLayer2 = tf.contrib.layers.fully_connected( fcMeanLayer1, num_outputs = self.neuronsFc2, weights_initializer = weightsInitFcMean2, biases_initializer = biasInitFcMean2)

            weightsInitFcVariance1 = tf.truncated_normal_initializer( stddev = math.sqrt( 2 / (self.lstmUnits + self.neuronsFc1) ) )
            biasInitFcVariance1 = tf.truncated_normal_initializer( stddev = math.sqrt(2 / self.neuronsFc1) )
            fcVarianceLayer1 = tf.contrib.layers.fully_connected( lastSequenceSample, num_outputs = self.neuronsFc1, weights_initializer = weightsInitFcVariance1, biases_initializer = biasInitFcVariance1)

            weightsInitFcVariance2 = tf.truncated_normal_initializer( stddev = math.sqrt( 2 / (self.neuronsFc1 + self.neuronsFc2) ) )
            biasInitFcVariance2 = tf.truncated_normal_initializer( stddev = math.sqrt(2 / self.neuronsFc2) )
            fcVarianceLayer2 = tf.contrib.layers.fully_connected( lastSequenceSample, num_outputs = self.neuronsFc2, weights_initializer = weightsInitFcVariance2, biases_initializer = biasInitFcVariance2)

            # output heads for mean and variance
            weightsInitOutputMean = tf.truncated_normal_initializer( stddev = math.sqrt( 2 / (self.neuronsFc2 + self.nOutputs) ) )
            biasInitOutputMean = tf.constant_initializer(0)
            outputMean = tf.contrib.layers.fully_connected( fcMeanLayer2, num_outputs = self.nOutputs, activation_fn = None, weights_initializer = weightsInitOutputMean, biases_initializer = biasInitOutputMean)

            weightsInitOutputVariance = tf.truncated_normal_initializer( stddev = math.sqrt( 2 / (self.neuronsFc2 + self.nOutputs) ) )
            biasInitOutputVariance = tf.constant_initializer(0)
            outputVariance = tf.contrib.layers.fully_connected( fcVarianceLayer2, num_outputs = self.nOutputs, activation_fn = tf.sigmoid, weights_initializer = weightsInitOutputVariance, biases_initializer = biasInitOutputVariance)

            # wrapped outputs
            self.output = [outputMean, outputVariance]

class Batch:
    '''
    Batch of states, actions, rewards, next states and done state.

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
        ''' checks if the batch is filled completely

        OUT         (bool)      returns True if the number of entries is equal to the defined batch size '''

        if len(self.actions) >= self.size:
            return True
        else:
            return False
