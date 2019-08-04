from Sampler import *
import Rewarder
import ActionSpace

class Environment:
    '''
    The representation of the environment we are trying to train the agent on.
    It returns the current observations, trading state, reward and information
    if the episode is finished. It's the interface to provide random datasets

    INPUTS:
        dataPath           path to the data we want to sample from
        windowLength       window length of the dataset we move over the sample
    '''

    def __init__(self, dataPath, windowLength):
        self.SAMPLE_LENGTH = 720
        self.windowLength = windowLength
        self.sampler = Sampler(dataPath)
        self.actionSpace = ActionSpace.SeparatedNets()
        self.rewarder = Rewarder.SeparatedNetsSingleReward()
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
        observation = self.dataset.__next__()
        isDone = self.dataset.isLastWindow() or self.actionSpace.noActionsPossible()

        return observation, reward, tradingState, isDone
