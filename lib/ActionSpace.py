from abc import ABC, abstractmethod

class ActionSpace(ABC):
    '''
    The Actions Space is an abstract class for wrapping actions.

    It provides the mapping of the current trading state to the possible actions
    and the transitions from one action to the resulting trading state.
    When performing an action via the "do"-method, the resulting trading state
    is returned.

    '''

    def __init__(self):
        super().__init__()

    @property
    def tradingState(self):
        pass

    @property
    def possibleActions(self):
        pass

    @property
    def stateTransition(self):
        pass

    def getTradingState(self):
        ''' returns the current trading state as a verbal expression

            OUT                 (string)      current trading state'''

        return self.tradingState

    def isActionPossible(self, action):
        ''' check if performing the given action is possible

            OUT             (bool)      result if action is possible or not'''

        possibleActions = self._getPossibleActions()
        return action in possibleActions

    def noActionsPossible(self):
        ''' returns true if no further action is possible

            OUT         (bool)      result if no action is possible'''

        return len( self._getPossibleActions() ) == 0

    def _getPossibleActions(self):
        ''' list of all possible actions

            OUT         (list)      all possible actions as verbal expressions'''

        return self.possibleActions[self.tradingState]

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def do(self, action):
        pass

class SeparatedNets(ActionSpace):

    def __init__(self):
        super().__init__()
        self._tradingState = "observe" # observe -> open_position -> close_position
        self._possibleActions = {"observe":["keep_observing", "open_long", "open_short"], "open_position":["keep_position", "close_position"], "closed_position":[]} # {tradingState:[actions]}
        self._stateTransition = {"keep_observing":"observe", "open_long":"open_position", "open_short":"open_position", "keep_position":"open_position", "close_position":"closed_position"}

    @property
    def tradingState(self):
        return self._tradingState

    @tradingState.setter
    def tradingState(self, value):
        self._tradingState = value

    @property
    def possibleActions(self):
        return self._possibleActions

    @possibleActions.setter
    def possibleActions(self, value):
        pass

    @property
    def stateTransition(self):
        return self._stateTransition

    @stateTransition.setter
    def stateTransition(self, value):
        pass

    def reset(self):
        ''' resets the trading state to the beginning state '''

        self.tradingState = "observe"

    def do(self, action):
        ''' returns the new trading state after performing the given action. raises an exception if
            the action is not possible

            OUT         (string)        new trading state as verbal expression'''

        if not self.isActionPossible(action): raise ValueError("action is not possible")
        self.tradingState = self.stateTransition[action]

        return self.tradingState
