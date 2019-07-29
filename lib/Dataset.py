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
        self.iterPosition = 0

    def __iter__(self):
        ''' initializes the iteration '''

        self.iterPosition = self.windowLength - 1 # iterPosition points to the index of the end point of the window

        return self

    def __next__(self):
        ''' performs iteration step. each iteration step we move the windows of  obtained data one data point forward

            OUT                 (pandas.dataframe)      dataframe window of the current iteration'''

        self.iterPosition += 1
        if self.iterPosition > self.length: raise StopIteration

        start = self.iterPosition - self.windowLength
        end = self.iterPosition

        return self.data[start:end]

    def getSubjects(self):
        ''' returns a list of the available subjects of the trading data

        OUT             (list)      standardized names of the available subjects of each data symbol, e.g. ["price_ask_open", "macd_line"] '''

        return self.data.columns.to_list()

    def getLength(self):
        ''' returns the length of the complete dataset

        OUT                     (int)      length of the dataset '''

        return self.length

    def getPosition(self):
        ''' returns the sample number of the dataset where the recent end position of the window is located

        OUT                     (int)      current iterator position '''

        return self.iterPosition
