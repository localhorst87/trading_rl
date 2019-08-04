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

    def reset(self):
        ''' resetting the properties. Use this method when starting a new episode '''

        self.dataset = None
        self.tradeType = None
        self.sampleOpen = None
        self.sampleClose = None

    @property
    def rewardFunctionMap(self):
        pass

    @abstractmethod
    def getReward(self, dataset, action):
        pass

class SeparatedNetsSingleReward(Rewarder):
    '''
    Rewarder for actions of separated networks (one network where to decide when to open
    a position and one for deciding when to close a position) that rewards only the last
    action for each net (opening and closing). All actions where we observe or keep the
    position are rewarded with 0.
    '''

    def __init__(self):
        super().__init__()
        self.WEIGHT_FACTOR_FORW = 0.97
        self.WEIGHT_FACTOR_BACK = 0.75
        self._rewardFunctionMap = {"keep_observing":self._rewardKeepObserving, "open_long":self._rewardOpenLong, "open_short":self._rewardOpenShort, "keep_position":self._rewardKeepPosition, "close_position":self._rewardClosePosition}

    @property
    def rewardFunctionMap(self):
        return self._rewardFunctionMap

    @rewardFunctionMap.setter
    def rewardFunctionMap(self, value):
        pass

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
        ''' returns the reward for holding a position '''

        return 0

    def _rewardClosePosition(self):
        ''' returns the reward closing a position '''

        self.sampleClose = self.dataset.getPosition()

        if self.tradeType == "long":
            priceOpen = self.dataset.data["price_ask_close"][self.sampleOpen]
            priceClose = self.dataset.data["price_bid_close"][self.sampleClose]
            typeFactor = 1
        elif self.tradeType == "short":
            priceOpen = self.dataset.data["price_bid_close"][self.sampleOpen]
            priceClose = self.dataset.data["price_ask_close"][self.sampleClose]
            typeFactor = -1
        else:
            raise ValueError("no valid open position existing")

        tradeReturn = typeFactor * (priceClose - priceOpen) / priceOpen

        return scaledSigmoid(tradeReturn, min = -1, max = 1, scaleFactor = 115)

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
