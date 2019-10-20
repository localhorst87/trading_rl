from abc import ABC, abstractmethod
import numpy as np
import scipy

class Rewarder(ABC):
    '''
    Interface for Rewarder classes. A Rewarder class calculates the reward for
    each action. Depending on the type of action space of type of reward calculation
    different implementations for this interface are used.

    The quality and stability of a reinforcement learning network for trading
    does highly depend on the reward function!
    '''

    def __init__(self):
        super().__init__()
        self.dataset = None
        self.tradeType = None # "long" / "short"
        self.sampleOpen = None
        self.sampleClose = None
        self.rewardFunctionMap = {"keep_observing":self._rewardKeepObserving, "open_long":self._rewardOpenLong, "open_short":self._rewardOpenShort, "keep_position":self._rewardKeepPosition, "close_position":self._rewardClosePosition}

    def reset(self):
        ''' resetting the properties. Use this method when starting a new episode '''

        self.dataset = None
        self.tradeType = None
        self.sampleOpen = None
        self.sampleClose = None

    def getReward(self, dataset, action):
        ''' returns the reward of the given action. The correct method for the
        action is selected according to the rewardFunctionMap

        IN      dataset     (Dataset)       current used Dataset
        IN      action      (string)        performed action
        OUT                 (float)         reward of the action '''

        self.dataset = dataset
        rewardFunctionHandle = self.rewardFunctionMap[action]
        reward = rewardFunctionHandle()

        return reward

class WeightedMaxMinOpeningRewarder(Rewarder):
    '''
        Rewarder for actions of separated networks (one network where to decide when to open
        a position and one for deciding when to close a position) that rewards only the last
        action for each net (opening and closing). All actions where we observe or keep the
        position are rewarded with 0.

        This Rewarder provides rewards for training of an opening agent only.
        The reward for opening a position is calculated via a time and scale weighted
        maximum and minimum future price.
    '''

    def __init__(self):
        super().__init__()
        self.SAMPLES_FORWARD_OBSERVE = 200

    def _rewardKeepObserving(self):
        ''' returns the reward for observing the market before opening a position '''

        return 0

    def _rewardOpenLong(self):
        ''' returns the reward for opening a long position '''

        self.tradeType = "long"
        self.sampleOpen = self.dataset.getPosition()
        weightedReturns = self._calcWeightedReturns()
        expectedWeightedReturn = self._calcRewardFromWeightedReturns(weightedReturns)

        return expectedWeightedReturn

    def _rewardOpenShort(self):
        ''' returns the reward for opening a long position '''

        self.tradeType = "short"
        self.sampleOpen = self.dataset.getPosition()
        weightedReturns = self._calcWeightedReturns()
        expectedWeightedReturn = self._calcRewardFromWeightedReturns(weightedReturns)

        return expectedWeightedReturn

    def _rewardKeepPosition(self):
        pass

    def _rewardClosePosition(self):
        pass

    def _calcWeightedReturns(self):
        ''' calculates time weighted price change in percentages relative to the open position.

            OUT                 (list)      time and scale weighted returns
        '''

        if self.tradeType == "long":
            filteredPrice = self._getFilteredPrice("price_ask_close")
            typeFactor = 1
        elif self.tradeType == "short":
            filteredPrice = self._getFilteredPrice("price_bid_close")
            typeFactor = -1
        else:
            raise ValueError("no valid open position existing")

        start = self.sampleOpen
        end = start + self.SAMPLES_FORWARD_OBSERVE
        tradeReturns =  typeFactor * (filteredPrice[start:end] - filteredPrice[start]) / filteredPrice[start]

        timeWeights = self._getTimeWeights( len(tradeReturns) )
        timeWeightedReturns = tradeReturns * timeWeights

        return timeWeightedReturns

    def _calcRewardFromWeightedReturns(self, weightedReturns):
        ''' calculates the reward from the time weighted returns using the values of the maximum
            and minimum values and scales it with the volatility level

            IN      weightedReturns     (list)      time and scale weighted returns
            OUT                         (float)     expected weighted return
        '''

        minValue = min(weightedReturns)
        maxValue = max(weightedReturns)
        volatilityLevel = self._getVolatilityLevel()

        expectedValue = (maxValue + minValue) / volatilityLevel

        return expectedValue

    def _getTimeWeights(self, length):
        ''' calculates time weights for the trade returns. The further away the sample from the
            opening position the lower the weight factor

            IN      length      (int)       length of the weightedReturns
            OUT                 (list)
        '''

        timeWeight = [scaledSigmoid(x, scaleFactor = 0.75) for x in np.linspace(5, -5, self.SAMPLES_FORWARD_OBSERVE)]

        return timeWeight[:length]

    def _getFilteredPrice(self, subject):
        ''' returns a low pass filtered price, according to the given subject.
            Low pass filtering prevents from overfitting and supports training stability

            IN      subject         (string)        standardized name of available subjects of a data symbol, e.g. "price_ask_open"
            OUT     priceFiltered   (np.array)      low pass filtered price values
        '''

        price = self.dataset.data[subject].values
        b, a = scipy.signal.butter(4, 0.15)
        priceFiltered = scipy.signal.filtfilt(b, a, price)

        return priceFiltered

    def _getVolatilityLevel(self):
        ''' returns the a level to norm returns over many forex pairs. It's calculated
            via the mean standard deviation compared to the mean price over the last 480
            samples. Make sure to ingest these indicator data when using this rewarder!

            OUT             (float)     volatilityLevel
        '''

        stddev = self.dataset.data["stddev_480"].mean()
        meanPrice = self.dataset.data["sma_480"].mean()

        return stddev/meanPrice

class CumulativeWeightedOpeningRewarder(Rewarder):
    '''
    Rewarder for actions of separated networks (one network where to decide when to open
    a position and one for deciding when to close a position) that rewards only the last
    action for each net (opening and closing). All actions where we observe or keep the
    position are rewarded with 0.

    This Rewarder provides rewards for training of an opening agent only.
    The reward for opening a position is calculated via cumulative weighted price
    changes of future prices and weighted missed return.
    '''

    def __init__(self):
        super().__init__()
        self.SAMPLES_FORWARD_OBSERVE = 120

    def _rewardKeepObserving(self):
        ''' returns the reward for observing the market before opening a position '''

        return 0

    def _rewardOpenLong(self):
        ''' returns the reward for opening a long position '''

        self.tradeType = "long"
        self.sampleOpen = self.dataset.getPosition()

        reward = self._calcExpectedReturn() / self._calcVolatilityLevel()

        return reward

    def _rewardOpenShort(self):
        ''' returns the reward for opening a short position '''

        self.tradeType = "short"
        self.sampleOpen = self.dataset.getPosition()

        reward = self._calcExpectedReturn() / self._calcVolatilityLevel()

        return reward

    def _rewardKeepPosition(self):
        pass

    def _rewardClosePosition(self):
        pass

    def _getTimeWeights(self, length):
        ''' calculates time weights for the trade returns. The further away the sample from the
            opening position the lower the weight factor

            IN      length      (int)       length of the weightedReturns
            OUT                 (list)
        '''

        timeWeight = [scaledSigmoid(x, scaleFactor = 0.75) for x in np.linspace(5, -5, self.SAMPLES_FORWARD_OBSERVE)]

        return timeWeight[:length]

    def _getFilteredPrice(self, subject):
        ''' returns a low pass filtered price, according to the given subject.
            Low pass filtering prevents from overfitting and supports training stability

            IN      subject         (string)        standardized name of available subjects of a data symbol, e.g. "price_ask_open"
            OUT     priceFiltered   (np.array)      low pass filtered price values
        '''

        price = self.dataset.data[subject].values
        b, a = scipy.signal.butter(4, 0.15)
        priceFiltered = scipy.signal.filtfilt(b, a, price)

        return priceFiltered

    def _calcExpectedReturn(self):
        ''' calculates the expected return after opening a position. The expected
        return does take in account the future developement of the price. It is
        weighted according to the distance of the opening bar. So if the price
        developes well right after opening the position it will result in a higher
        expected return than if the price developes well later.

        OUT         (float)         weighed expected return '''

        if self.tradeType == "long":
            filteredPrice = self._getFilteredPrice("price_ask_close")
            typeFactor = 1
        elif self.tradeType == "short":
            filteredPrice = self._getFilteredPrice("price_bid_close")
            typeFactor = -1
        else:
            raise ValueError("no valid open position existing")

        start = self.sampleOpen + 1
        end = start + self.SAMPLES_FORWARD_OBSERVE

        futureReturns = [typeFactor * (filteredPrice[i] - filteredPrice[i-1]) / filteredPrice[i-1] for i in range(start, end)]
        timeWeights = self._getTimeWeights( len(futureReturns) )
        expectedReturn = sum( [futureReturns[i] * timeWeights[i] for i in range( len(futureReturns) )] )

        return expectedReturn

    def _calcVolatilityLevel(self):
        ''' returns the a level to norm returns over many forex pairs. It's calculated
            via the mean standard deviation compared to the mean price over the last 480
            samples. Make sure to ingest these indicator data when using this rewarder!

            OUT             (float)     volatilityLevel
        '''

        stddev = self.dataset.data["stddev_480"].mean()
        meanPrice = self.dataset.data["sma_480"].mean()

        return stddev/meanPrice

def scaledSigmoid(x, **kwargs):
    ''' using a scaled sigmoid function to clip a value between a defined minimum
        and maximum

        IN              x           (float)     value to clip
        IN      **kwarg max         (int)       maximum clipping value (default: 1)
        IN      **kwarg min         (int)       minimum clipping value (default: 0)
        IN      **kwarg scaleFactor (float)     inner scaling factor of the sigmoid function (default: 1)
        IN      **kwarg offset      (float)     inner offset of the sigmoid function (default: 0)'''

    max = kwargs["max"] if "max" in kwargs.keys() else 1
    min = kwargs["min"] if "min" in kwargs.keys() else 0
    a = kwargs["offset"] if "offset" in kwargs.keys() else 0
    k = kwargs["scaleFactor"] if "scaleFactor" in kwargs.keys() else 1

    if min > max: max, min = min, max # swap values if min is greater than max value

    return (max-min) / ( 1 + np.exp(-k*x + a) ) + min
