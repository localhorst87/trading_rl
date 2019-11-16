from DataProvider import Sampler
import Rewarder
import ActionSpace
import numpy as np
import pickle
import collections

class Environment:
    '''
    The representation of the environment we are trying to train the agent on.
    It returns the current observations, trading state, reward and information
    if the episode is finished. It's the interface to provide random datasets

    INPUTS:
        dataPath           path to the data we want to sample from
        windowLength       window length of the dataset we move over the sample
    '''

    def __init__(self, dataPath, windowLength, actionSpace, rewarder):
        self.SAMPLE_LENGTH = 720
        self._windowLength = windowLength # never set self._windowLength during the iteration, only self.windowLength, as it will call the method to adjust the dataset
        self.sampler = Sampler(dataPath)
        self.actionSpace = actionSpace
        self.rewarder = rewarder
        self.dataset = None

    def reset(self):
        ''' initiates a new episode by requesting a new dataset

            OUT         (pandas.DataFrame)      initial window of the Dataset'''

        self.dataset = self.sampler.getRandomDataset(self.windowLength, self.SAMPLE_LENGTH)
        self.dataset.__iter__()
        self.actionSpace.reset()
        self.rewarder.reset()

        return self.dataset.__next__()

    def step(self, action):
        ''' performs a step. That means perform an action and return the current observation, reward and trading state

            OUT     observation     (pandas.DataFrame)      current window of the Dataset
            OUT     reward          (float)                 reward of the last action
            OUT     tradingState    (string)                current trading situation, e.g. "open" for an open position
            OUT     isDone          (bool)                  environment finish state
         '''

        tradingState = self.actionSpace.do(action)
        reward = self.rewarder.getReward(self.dataset, action)
        nextObservation = self.dataset.__next__()
        isDone = self.dataset.isLastWindow() or self.actionSpace.noActionsPossible()

        return nextObservation, reward, tradingState, isDone

    @property
    def windowLength(self):
        return self._windowLength

    @windowLength.setter
    def windowLength(self, value):
        self._windowLength = value
        self.dataset.changeWindowLength(value)

class ReplayMemory:
    '''
    Memory of experienced samples. The maximum size of the memory is bound to the given size.
    Can return a random permutation of samples

    INPUTS:
        size    (int)       maximum nubmer of samples to collect
    '''

    def __init__(self, size):
        self.memory = collections.deque(maxlen = size)

    def save(self, path):
        ''' saves the memory on the desired path

            IN      path    (string)    relative or absolute file path to save the memory
        '''

        fileHandler = open(path, 'wb')
        pickle.dump(self.memory, fileHandler)

    def load(self, path):
        ''' loads a saved memory from HDD

            IN      path    (string)    relative or absolute file of the saved memory
        '''

        fileHandler = open(path, 'rb')
        self.memory = pickle.load(fileHandler)

    def getSamples(self, batchSize):
        ''' returns random samples of the replay memory.
            Throws an exception if the number of samples in the memory is smaller than the requested batch size

            IN      batchSize   (int)       number of random samples to return
            OUT                 (list)      random sample list
        '''
        if len(self.memory) < batchSize: raise ValueError("Not enough samples to create a batch of the desired size")

        permutation = np.random.permutation( len(self.memory) )[0:batchSize]
        randomSamples = np.array(self.memory)[permutation]

        return randomSamples.tolist()

    def add(self, sample):
        ''' adds a sample to the memory. If the memory size is greater than the maximum defined size,
            the oldest sample will be kicked out

            IN      sample      (Sample)    sample of state, action, reward, next state, done
        '''

        self.memory.append(sample)

    def getLength(self):
        ''' returns the number of samples of the replay memory

            OUT         (int)       number of samples
        '''

        return len(self.memory)

    def _isEmpty(self):
        ''' returns True if the memory is empty '''

        return self.getLength() == 0

class Sample:
    '''
    wrapper for state, action, reward, next state and done observations from the environment
    '''

    def __init__(self):
        self.state = None
        self._action = None
        self._reward = None
        self.nextState = None
        self._done = None

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, value):
        if not isinstance(value, int): raise ValueError("action must be an integer number")
        self._action = value

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, value):
        if not isinstance(value, (int, float) ): raise ValueError("reward must be numeric")
        self._reward = value

    @property
    def done(self):
        return self._done

    @done.setter
    def done(self, value):
        if not isinstance(value, bool): raise ValueError("done must be a boolean")
        self._done = value

    def isSet(self):
        ''' returns true if at least a state, action and reward are set '''

        isSet = self.state is not None and self.action is not None and self.reward is not None

        return isSet
