from abc import ABC, abstractmethod
import numpy as np

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
        reward = self._calcRewardFromWeightedReturns(weightedReturns)

        return reward

    def _rewardOpenShort(self):
        ''' returns the reward for opening a long position '''

        self.tradeType = "short"
        self.sampleOpen = self.dataset.getPosition()
        weightedReturns = self._calcWeightedReturns()
        reward = self._calcRewardFromWeightedReturns(weightedReturns)

        return reward

    def _rewardKeepPosition(self):
        pass

    def _rewardClosePosition(self):
        pass

    def _calcWeightedReturns(self):
        ''' calculates the time and scale weighted price change in percentages relative to the
            open position. Important: First weighting with the time weights and as second with
            the scale weights, not vice versa!

            OUT                 (list)      time and scale weighted returns
        '''

        if self.tradeType == "long":
            price = self.dataset.data["price_ask_close"]
            typeFactor = 1
        elif self.tradeType == "short":
            price = self.dataset.data["price_bid_close"]
            typeFactor = -1
        else:
            raise ValueError("no valid open position existing")

        start = self.sampleOpen + 1
        end = start + self.SAMPLES_FORWARD_OBSERVE
        tradeReturns =  100 * typeFactor * (price[start:end] - price[start]) / price[start] # percentage

        timeWeights = self._getTimeWeights( len(tradeReturns) )
        timeWeightedReturns = tradeReturns * timeWeights
        weightedReturns = [self._scaleWeightFunction(x) for x in timeWeightedReturns.values]

        return weightedReturns

    def _calcRewardFromWeightedReturns(self, weightedReturns):
        ''' calculates the reward from the weighted returns using the values of the maximum
            and minimum values of the weighted returns

            IN      weightedReturns     (list)      time and scale weighted returns
            OUT                         (float)     reward
        '''

        minValue = min(weightedReturns)
        maxValue = max(weightedReturns)
        minPosition = np.argmin(weightedReturns)
        maxPosition = np.argmax(weightedReturns)

        expectedValue = maxValue + minValue

        return scaledSigmoid(expectedValue, min = -1, max = 1, scaleFactor = 0.85)

    def _getTimeWeights(self, length):
        ''' calculates time weights for the trade returns. The further away the sample from the
            opening position the lower the weight factor

            IN      length      (int)       length of the weightedReturns
            OUT                 (list)
        '''

        timeWeight = [scaledSigmoid(x, scaleFactor = 0.75) for x in np.linspace(5, -5, self.SAMPLES_FORWARD_OBSERVE)]

        return timeWeight[:length]

    def _scaleWeightFunction(self, tradeReturn):
        ''' calculates the scale weighted trade return of a trade return: Higher values are
            rewarded with a gaining factor, lower values are damped.
            The damping and gaining is stronger for negative returns than for positive returns.

            IN      tradeReturn     (float)     trade return
            OUT                     (float)     scale weighted return
        '''

        power = 1.4 if tradeReturn >= 0 else 1.8

        return np.sign(tradeReturn) * abs(tradeReturn)**power

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
        self.WEIGHT_FACTOR_FORW = 0.99
        self.WEIGHT_FACTOR_BACK = 0.75

    def _rewardKeepObserving(self):
        ''' returns the reward for observing the market before opening a position '''

        return 0

    def _rewardOpenLong(self):
        ''' returns the reward for opening a long position '''

        self.tradeType = "long"
        self.sampleOpen = self.dataset.getPosition()

        missedReturnPerBar = self._calcMissedReturn(self.WEIGHT_FACTOR_BACK)
        expectedReturnPerBar = self._calcExpectedReturn(self.WEIGHT_FACTOR_FORW)
        reward = expectedReturnPerBar - missedReturnPerBar

        return scaledSigmoid(reward, min = -1, max = 1, scaleFactor = 13000)

    def _rewardOpenShort(self):
        ''' returns the reward for opening a short position '''

        self.tradeType = "short"
        self.sampleOpen = self.dataset.getPosition()

        missedReturnPerBar = self._calcMissedReturn(self.WEIGHT_FACTOR_BACK)
        expectedReturnPerBar = self._calcExpectedReturn(self.WEIGHT_FACTOR_FORW)
        reward = expectedReturnPerBar - missedReturnPerBar

        return scaledSigmoid(reward, min = -1, max = 1, scaleFactor = 13000)

    def _rewardKeepPosition(self):
        pass

    def _rewardClosePosition(self):
        pass

    def _calcMissedReturn(self, weightFactor):
        ''' calculates the missed return When opening a position. The missed
        return is the return that you missed for waiting too long - so actually
        the difference to the local minimum(long) / maximum(short) before opening
        the position.

        Here the missed return is calculated as a weighted return: The farther
        away the sample, the less it contributes to the missed return. The value
        for the missed return is always greater or equals zero.

        OUT         (float)         weighted missed return '''

        if self.tradeType == "long":
            price = self.dataset.data["price_ask_close"]
            typeFactor = 1
        elif self.tradeType == "short":
            price = self.dataset.data["price_bid_close"]
            typeFactor = -1
        else:
            raise ValueError("no valid open position existing")

        missedReturns = [(price[i] - price[i-1]) / price[i-1] for i in range(self.sampleOpen, self.dataset.windowLength - 1, -1)]
        cumulativeWeightedMissedReturn = sum( [missedReturns[i] * weightFactor**i for i in range( len(missedReturns) )] )
        weightsSum = sum( [weightFactor**i for i in range( len(missedReturns) )] )
        missedReturnPerBar = max(0, typeFactor * cumulativeWeightedMissedReturn / weightsSum)

        return missedReturnPerBar

    def _calcExpectedReturn(self, weightFactor):
        ''' calculates the expected return after opening a position. The expected
        return does take in account the future developement of the price. It is
        weighted according to the distance of the opening bar. So if the price
        developes well right after opening the position it will result in a higher
        expected return than if the price developes well later.

        OUT         (float)         weighed expected return '''

        if self.tradeType == "long":
            price = self.dataset.data["price_ask_close"]
            typeFactor = 1
        elif self.tradeType == "short":
            price = self.dataset.data["price_bid_close"]
            typeFactor = -1
        else:
            raise ValueError("no valid open position existing")

        futureReturns = [(price[i] - price[i-1]) / price[i-1] for i in range(self.sampleOpen + 1, self.dataset.length)]
        cumulativeWeightedReturn = sum( [futureReturns[i] * weightFactor**i for i in range( len(futureReturns) )] )
        weightSum = sum( [weightFactor**i for i in range( len(futureReturns) )] )
        expectedReturnPerBar = typeFactor * cumulativeWeightedReturn / weightSum

        return expectedReturnPerBar

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
