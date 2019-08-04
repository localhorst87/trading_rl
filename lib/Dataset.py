class Dataset:
    '''
    The Dataset class is a wrapper for a data sample. It loops the data by returning
    a window of the last observed datapoints.
    Every step we iterate over the dataset the window is shifting one datapoint forward.

    Dataframes are expected to be consisting of standardized pandas series

    INPUTS:
        dataframe       the complete pandas.dataframe sample of one trading data symbol
        windowLength    the length of the moving window
    '''

    def __init__(self, dataframe, windowLength):
        self.data = dataframe
        self.length = len(dataframe)
        self.windowLength = windowLength
        self.nIteration = 0

    def __iter__(self):
        ''' initializes the iteration '''

        self.nIteration = 0

        return self

    def __next__(self):
        ''' performs iteration step. each iteration step we move the windows of obtained data one data point forward
            the first iteration returns the first window (no window forward shifting yet!)

            OUT         (pandas.dataframe)      dataframe window of the current iteration'''

        self.nIteration += 1
        if self.getPosition() == self.length: raise StopIteration

        start = self.nIteration - 1
        end = start + self.windowLength

        return self.data[start:end]

    def getSubjects(self):
        ''' returns a list of the available subjects of the trading data

        OUT             (list)      standardized names of the available subjects of each data symbol, e.g. ["price_ask_open", "macd_line"] '''

        return self.data.columns.to_list()

    def getPosition(self):
        ''' returns the sample number of the dataset where the recent end position of the window is located
            example after the 1st iteration and window length 5, the index of the windows last point is 4

        OUT             (int)      current index of the last windows sample '''

        return (self.nIteration - 1) + (self.windowLength - 1)

    def isLastWindow(self):
        ''' returns if the window that was returned by the last iteration is the last window in the dataset

        OUT             (bool)      if true, then window was the last window. no more iterations possible'''

        return self.getPosition() == self.length - 1
