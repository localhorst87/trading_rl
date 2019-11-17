from abc import ABC, abstractmethod
from Environment import *
from NeuralNetworks import *
from DataProvider import *
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
        self.graph = tf.Graph()
        self.session = tf.Session(graph = self.graph)
        self.trainable = True
        self.loadModelPath = None
        self.episode = 0

    def __del__(self):
        self.session.close()

    def _episodeWrapper(runEpisodeFunc):
        ''' decorator for runEpisode. While in the runEpisode method should be defined the arbitrary stuff as handling of the states, actions, rewards, etc.
            the _episodeWrapper method performs the procedure around that, as incrementing the step, performing the training etc.
        '''

        def decoratorWrapper(self):

            if self.episode == 0:
                if self.loadModelPath is None:
                    with self.networkHandler.graph.as_default(): tf.global_variables_initializer().run(session = self.session) # init variables
                else:
                    self.networkHandler.load(self.loadModelPath) # load variables

            try:
                reward, lastAction = runEpisodeFunc(self) # here the episode itself is performed

            except KeyboardInterrupt:
                raise

            except:
                print("error occured")
                reward = None
                lastAction = None

            self.episode += 1

            if self.trainable and self.trainingActive and self.episode % self.stepSizeTraining == 0:
                batch = self._createBatch()
                additionalParameters = self._getAdditionalTrainParameters()
                loss = self.networkHandler.train(batch, self.learningRate, **additionalParameters)
            else:
                loss = None

            return reward, lastAction, loss

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
    def trainingActive(self):
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

        The OpeningAgent uses sequence data that is forwarded to a convolutional neural network with fully connected layers.
        It follows a stricty value based and deterministic policy. The training is performed via MonteCarlo
        sampling from a replay memory, as it does not use policy gradients.

        INPUTS:
            environment             (Environment)   The environment to sample and receive rewards from

        CONFIGURABLE PROPERTIES:
            batchSize               (int)           number of episodes inside a single training batch
            learningRate            (float)         learning Rate for training
            stepSistepSizeTraining  (int)           intervall of episodes each time a training will be executed
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
        self.replayMemory = ReplayMemory(10000) # check memory consumption first before setting the length of the replay memory
        self.batchSize = 1000
        self._learningRate = 1e-3
        self._stepSizeTraining = 250
        self.episodeStartTraining = 10000
        self._trainingActive = False
        self._environment = environment
        self._minIterationsToOpen = 0 # wait this number of iterations until opening a position will be possible
        self._minStepsLeft = 200 # abort episode after this number of (or less) remaining datapoints
        self._possibleActions = self.environment.actionSpace.possibleActions["observe"]
        self._observeAction = self.environment.actionSpace.possibleActions["observe"][0]
        self._openingActions = self.environment.actionSpace.possibleActions["observe"][1:3]
        self._nOpeningActions = len(self._openingActions)
        self._nInputs = 3 # number of signals as input
        self._createNetworks()

    def _createNetworks(self):
        ''' creates a networkHandler according to the following configuration '''

        inputShape = [None, self._nInputs, self.environment.windowLength, 1]
        network = CnnNormalDistribution("observeNet", inputShape, self._nOpeningActions)
        self.networkHandler = NormalDistributionHandler(network, self.session, self.graph)

    @Agent._episodeWrapper
    def runEpisode(self):
        ''' executes exactly one single episode '''

        if self.episode >= self.episodeStartTraining: self.trainingActive = True

        observation = self.environment.reset()
        state = self._createInput(observation)
        tradingState = "observe"

        while tradingState == "observe" and self.environment.dataset.getRemainingSteps() > self._minStepsLeft:

            action = self._getAction(state) # action string, e.g. "keep_observing"
            nextObservation, reward, tradingState, _ = self.environment.step(action)
            nextState = self._createInput(nextObservation)

            if action != self._observeAction and self.trainable:
                sample = self._createSample(state, action, reward)
                self.replayMemory.add(sample)

            state = nextState

        return reward, action

    def _getAdditionalTrainParameters(self):
        ''' no additional parameters needed. Returns an empty (keyword argument) dictionary '''

        return dict()

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

            IN      state           (list)      state vectors before action, transformed from the raw environmental observation
            IN      action          (string)    action expression of the performed action
            IN      reward          (float)     last reward returned by the environment
            OUT     sample          (Sample)    standardized sample of one step
        '''

        actionNumber = self._openingActions.index(action)

        sample = Sample()
        sample.state = state
        sample.reward = reward
        sample.action = actionNumber

        return sample

    def _createInput(self, observation):
        ''' creates an network input ("state") of the observation from the environment

            IN      observation     (Pandas.Dataframe)      observation of the environment
            OUT                     (list)                  states that can be used as an input for the LSTM network
        '''

        macdLineFast = 100 * (observation["sma_6"]-observation["sma_18"]) / observation["sma_480"]
        macdLineFast = np.expand_dims(macdLineFast.values, axis = -1).tolist()
        macdLineMedium = 100 * (observation["sma_12"]-observation["sma_36"]) / observation["sma_480"]
        macdLineMedium = np.expand_dims(macdLineMedium.values, axis = -1).tolist()
        macdLineSlow = 100 * (observation["sma_36"]-observation["sma_84"]) / observation["sma_480"]
        macdLineSlow = np.expand_dims(macdLineSlow.values, axis = -1).tolist()

        return [macdLineFast, macdLineMedium, macdLineSlow] # return lists and not numpy arrays as numpy uses an own specified memory allocator. Memory consumption can increase vast then when using replay memories!

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

    @property
    def trainingActive(self):
        return self._trainingActive

    @trainingActive.setter
    def trainingActive(self, value):
        self._trainingActive = value

class ClosingAgent(Agent):
    ''' Concrete Agent for estimating when to close a position.

        The ClosingAgent uses sequence data that is forwarded to a CNN with fully connected layers.
        It follows a stricty value based and deterministic policy. The training is performed via MonteCarlo
        sampling from a replay memory, as it does not uses policy gradients.

        INPUTS:
            environment             (Environment)   The environment to sample and receive rewards from

        CONFIGURABLE PROPERTIES:
            learningRate            (float)         learning Rate for training
            discountFactor          (float)         (0...1) discount factor for calculating the Q value
            minStepsKeepPosition    (int)           number of steps the position will be kept at least
            probabilityThreshold    (float)         the threshold the "close_position" probability must exceed to close the position when using a deterministic strategy

    '''

    def __init__(self, environment):
        super().__init__()
        self._learningRate = 1e-3
        self.discountFactor = 0.97
        self.replayMemory = ReplayMemory(50000) # use preprocessed samples only (this is on-policy)
        self.batchSize = 512
        self._stepSizeTraining = 20
        self._tradeType = None
        self._trainingActive = False
        self.strategy = "deterministic" # "deterministic" or "stochastic"
        self.probabilityThreshold = 0.60
        self._environment = environment # important: This is a reference, so one environment can be used sequentially in an opening and closing agent
        self._minStepsLeft = 1 # abort episode after this number of (or less) remaining steps to close the position
        self.minStepsKeepPosition = 5 # wait at least this number of steps until closing a position
        self._possibleActions = self.environment.actionSpace.possibleActions["open_position"]
        self._nClosingActions = len(self._possibleActions)
        self._episodeSamples = []
        self._nInputs = 3 # number of signals as input
        self._createNetworks()

    def _createNetworks(self):
        ''' creates the networkHandler for opening and closing network according to the following configuration.
            Opening network is created to not have a totally random configuration for the closing agent of opened
            positions.
        '''

        inputShape = [None, self._nInputs, self.environment.windowLength]
        network = LstmA2C("positionNet", inputShape, self._nClosingActions)
        self.networkHandler = A2CHandler(network, self.session, self.graph)

    @Agent._episodeWrapper
    def runEpisode(self):
        ''' executes exactly one single episode '''

        if self.replayMemory.getLength() >= self.batchSize: self.trainingActive = True

        self._setTradeTypeNumber() # sets the identifier of the current trade type (0 = long, 1 = short)
        observation = self.environment.dataset.getCurrentWindow()
        state = self._createInput(observation)
        tradingState = "open_position"
        self._episodeSamples = []

        while tradingState == "open_position" and self.environment.dataset.getRemainingSteps() > self._minStepsLeft:

            action = self._getAction(state, self.strategy) # action string ("keep_position" or "close_position")
            nextObservation, reward, tradingState, isDone = self.environment.step(action)
            nextState = self._createInput(nextObservation)

            sample = self._createSample(state, action, nextState, reward, isDone)
            self._episodeSamples.append(sample)

            state = nextState

        if len(self._episodeSamples) > 0: self._preprocessSamples()

        if tradingState == "closed_position":
            returnOnInvest = self.environment.rewarder.getReturnOfInvest()
            lastAction = action
        else:
            returnOnInvest = None
            lastAction = None

        return returnOnInvest, lastAction

    def _getAdditionalTrainParameters(self):
        ''' sets the trade type as an additional parameter for the training.

            OUT     additionalParameters    (dict)      (keyword argument) dictionary with additional parameters for training of the agent
        '''

        additionalParameters = {"tradeType": self._tradeType}

        return additionalParameters

    def _setTradeTypeNumber(self):
        ''' sets the trade type number (0 for long, 1 for short). This is important as the long
            and short closing network is architectually separated, so we need to know which network
            output to use for training!
        '''

        tradeTypeList = ["long", "short"]
        currentTradeType = self.environment.rewarder.tradeType

        self._tradeType = tradeTypeList.index(currentTradeType)

    def _getAction(self, state, strategy = "stochastic"):
        ''' returns the action to perform according to the given strategy:
            "stochastic": decide on given probabilities. "deterministic": take action with higher probability

            IN      state       (list)      state input vectors for the network
            IN      strategy    (string)    deterministic or non-deterministic strategy
            OUT     action      (string)    action expression of the chosen action
        '''

        holdTime = self.environment.dataset.getPosition() - self.environment.rewarder.positionOpen

        if holdTime < self.minStepsKeepPosition:
            action = "keep_position"
        elif strategy == "stochastic":
            action = self._getActionByProbability(state)
        elif strategy == "deterministic":
            action = self._getBestAction(state)
        else:
            raise ValueError("unknown strategy")

        return action

    def _getActionByProbability(self, state):
        ''' returns the best action according to the probabilities outputs of network

            IN      state       (list)      state input vectors for the network
            OUT     action      (string)    action expression of the best action
        '''

        actorProbs, _ = self._getEstimation(state)
        actionArray = np.random.choice(self._possibleActions, 1, p = actorProbs) # returns a numpy array
        action = actionArray[0] # extracts the action string from the array

        return action

    def _getBestAction(self, state):
        ''' returns the best action according to the probabilities outputs of network

            IN      state       (list)      state input vectors for the network
            OUT     action      (string)    action expression of the best action
        '''

        actorProbs, _ = self._getEstimation(state)
        bestAction = np.argmax(actorProbs)
        action = self._possibleActions[bestAction]

        if action == "close_position" and actorProbs[bestAction] >= self.probabilityThreshold:
            action = "close_position"
        else:
            action = "keep_position"

        return action

    def _getEstimation(self, state):
        ''' returns the estimation of actor and critic of the current state

            IN      state               (list)      state vectors from environment
            OUT     actorProbs          (list)      estimated probabilities for keep_position and close_position
            OUT     criticValue         (list)      estimated critic value for this state
        '''

        networkPrediction = self.networkHandler.predict(self.session, state)
        actorProbs = networkPrediction[0][self._tradeType]
        criticValue = networkPrediction[1][self._tradeType]

        return np.squeeze(actorProbs), np.squeeze(criticValue)

    def _createSample(self, state, action, nextState, reward, done):
        ''' creates a Sample of the last observation

            IN      state           (list)      state vectors before action, transformed from the raw environmental observation
            IN      action          (string)    action expression of the performed action
            IN      nextState       (list)      next state vectors after action, transformed from the raw environmental observation
            IN      reward          (float)     last reward returned by the environment
            IN      done            (bool)      indicates if the state AFTER performing the action is terminal or not
            OUT     sample          (Sample)    standardized sample of one step
        '''

        actionNumber = self._possibleActions.index(action)

        sample = Sample()
        sample.state = state
        sample.nextState = nextState
        sample.reward = reward
        sample.action = actionNumber
        sample.done = done

        return sample

    def _createInput(self, observation):
        ''' creates an network input ("state") of the observation from the environment

            IN      observation     (Pandas.Dataframe)      observation of the environment
            OUT                     (list)                  states that can be used as an input for the LSTM network
        '''

        windowLength = self.environment.dataset.windowLength

        if windowLength > self.episode + 1: # use only the data points after opening the position!
            nPoints = self.episode + 1
        else:
            nPoints = windowLength

        rsi = (observation["rsi_18"][windowLength-nPoints:] - 50) / 100
        rsi = smoothData(rsi, 3)
        rsi = rsi.values.tolist()

        macdLineSlow = 100 * (observation["sma_36"][:nPoints] - observation["sma_84"][:nPoints]) / observation["sma_480"][:nPoints]
        macdLineSlow = smoothData(macdLineSlow, 3)
        macdLineSlow = macdLineSlow.values.tolist()

        macdLineMedium = 100 * (observation["sma_12"][:nPoints] - observation["sma_36"][:nPoints]) / observation["sma_480"][:nPoints]
        macdLineMedium = smoothData(macdLineMedium, 3)
        macdLineMedium = macdLineMedium.values.tolist()

        # fill the rest of the data points with None values
        filler = [None] * (windowLength-nPoints)
        rsi.extend(filler)
        macdLineSlow.extend(filler)
        macdLineMedium.extend(filler)

        return [rsi, macdLineSlow, macdLineMedium] # return lists and not numpy arrays as numpy uses an own specified memory allocator. Memory consumption can increase vast then when using replay memories!

    def _preprocessSamples(self):
        ''' prepares the reversed Q value calculation for training '''

        lastSample = self._episodeSamples[-1]

        if lastSample.done:
            qValue = 0
        else:
            _, criticValueForecast = self._getEstimation(lastSample.nextState)
            qValue = criticValueForecast

        for sample in reversed(self._episodeSamples):
            qValue = sample.reward + self.discountFactor * qValue # start from the last sample and estimate Q value by reverse sampling the action history
            sample.reward = qValue # save Q value in the reward field of the sample
            self.replayMemory.add(sample)

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

    @property
    def trainingActive(self):
        return self._trainingActive

    @trainingActive.setter
    def trainingActive(self, value):
        self._trainingActive = value

class Batch:
    '''
    Batch of states, actions, rewards, next states and done state.

    INPUTS:
        size    (int)       size of the batch
    '''

    def __init__(self, size = -1):
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

        if len(self.actions) >= self.size and self.size != -1:
            return True
        else:
            return False
