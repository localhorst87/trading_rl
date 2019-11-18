import pickle
import random
import numpy as np
import os.path
import pandas as pd
from datetime import datetime, timedelta

class Sampler:
    '''
    The Sampler class is used for returning randomized snippets of financial data.

    The getRandomDataset method returns a Dataset object of a random data symbol for a random date
    according to the date boundaries. The windowLength determines the window length of the returned dataset.

    INPUTS:
        dataPath        The path of the financial data acquired via the QuantDataLoader
    '''

    def __init__(self, dataPath):
        self.MINIMUM_YEAR = 2008
        self.MAXIMUM_YEAR = 2015
        self.DAYS_MONTH = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
        self.data = {}
        self.resolution = 0
        self.symbolList = []
        self.subjectList = []
        self._load(dataPath)

    def getRandomDataset(self, windowLength, datasetLength):
        ''' creates a dataset of the loaded data of random date

        IN      windowLength    (integer)   defines the length of the window the dataset should return each iteration step
        OUT                     (Dataset)   a random Dataset of with the given parameters '''

        if windowLength >= datasetLength: raise ValueError("window length must be smaller than the dataset length")

        dataComplete = False

        while not dataComplete:
            forexSymbol = self._getRandomTradingSymbol()
            datetimeEnd = self._getRandomDatetime()
            dataframeCut = self._cutDataframe(self.data[forexSymbol], datetimeEnd, datasetLength) # cuts data by length
            if len(dataframeCut) < datasetLength or dataframeCut.isnull().values.any(): continue # price data not complete, try another dataset

            dataComplete = True

        return Dataset(dataframeCut, windowLength)

    def getDataInfo(self):
        ''' returns a list of the available financial data symbols and the related subjects of the dataset

        OUT     symbolList      (list)      standardized names of the available data symbols, e.g. ["EURUSD", "GBPUSD", "JPYAUD"]
        OUT     subjectList     (list)      standardized names of the available subjects of each data symbol, e.g. ["price_ask_open", "macd_line"] '''

        return self.symbolList, self.subjectList

    def _load(self, path):
        ''' loads data ingested and saved with QuantDataLoader class
        sets the priceData, indicatorData, resolution, symbolList and indicatorList from the loaded data

        IN      path    (string)    absolute or relative path of the trading data '''

        if not os.path.isfile(path): raise FileExistsError("data path not found")

        fileHandler = open(path, 'rb')
        importData = pickle.load(fileHandler)
        fileHandler.close()

        self.data = importData["data"]
        self.resolution = importData["resolution"]
        self.symbolList = list( self.data.keys() )
        self.subjectList = self.data[ self.symbolList[0] ].columns.to_list()

    def _cutDataframe(self, dataframe, datetimeEnd, length):
        ''' cuts the given dataframe to a dataframe of the given length. The last datapoint is determined by datetimeEnd

        IN      dataframe       (pandas.dataframe)      Pandas Dataframe to cut
        IN      datetimeEnd     (datetime)              Last datapoint of the cut dataframe
        IN      length          (int)                   length of the cut dataframe
        OUT                     (pandas.dataframe)      cut dataframe '''

        dateIndex = dataframe.index.values
        datetimeList = np.array( [pd.Timestamp(idx).to_pydatetime() for idx in dateIndex] )
        itemsToReturn = datetimeList <= datetimeEnd

        return dataframe.loc[itemsToReturn].tail(length) # returns the last <length> datapoints from <datetimeEnd> backwards

    def _getRandomTradingSymbol(self):
        ''' returns a random forex symbol of the available forex pairs

        OUT             (string)     ranom forex pair '''

        nSymbols = len(self.symbolList)
        randomForexNumber = random.randint(0, nSymbols-1)

        return self.symbolList[randomForexNumber]

    def _getRandomDatetime(self, maximumLimitation = True):
        ''' returns a random datetime by given bounaries in the properties.

        IN      maximumLimitation   (bool)      if set to False the last possible date is today, else the boundary conditions from the properties define the last possible date
        OUT                         (datetime)  randomized datetime '''

        datetimeNow = datetime.now()
        maximumYear = self.MAXIMUM_YEAR if maximumLimitation else datetimeNow.year
        randomYear = random.randint(self.MINIMUM_YEAR, maximumYear)
        randomMonth = random.randint(1, 12) if randomYear != datetimeNow.year else random.randint(1, datetimeNow.month)
        nDaysMonth = self.DAYS_MONTH[randomMonth] if not (self._isLeapYear(randomYear) and randomMonth == 2) else 29
        randomDay = random.randint(1, nDaysMonth)
        randomHour = random.randint(1, 23)
        randomMinute = random.randint(1, 59)

        randomDatetime = datetime(randomYear, randomMonth, randomDay, randomHour, randomMinute)

        return min(randomDatetime, datetimeNow)

    @staticmethod
    def _isLeapYear(year):
        ''' checks if the given year is a leap year

        IN      year   (int)      the year to check '''

        return True if year % 4 == 0 else False

class Dataset:
    '''
    The Dataset class is a wrapper for a data snippet. It loops the data by returning
    a window of the last observed datapoints.
    Every step we iterate over the dataset the window is shifting one datapoint forward.

    Dataframes are expected to be consisting of standardized pandas series

    INPUTS:
        dataframe       the complete pandas.dataframe of one trading data symbol
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

            OUT    currentWindow       (pandas.dataframe)      dataframe window of the current iteration
        '''

        self.nIteration += 1
        if self.getPosition() == self.length: raise StopIteration

        currentWindow = self.getCurrentWindow()

        return currentWindow

    def changeWindowLength(self, newLength):
        ''' changes the window size during the iteration, if possible

            IN      newLength       (int)       new window length
        '''

        if type(newLength) is not int: raise TypeError("new length must be of type integer")

        currentPosition = self.getPosition()
        if currentPosition < newLength - 1: raise ValueError("Conversion not possible. There are only %i data points in the past, %i required" % (currentPosition + 1, newLength) )

        self.nIteration = currentPosition - newLength + 2
        self.windowLength = newLength

    def getCurrentWindow(self):
        ''' returns the data of the current window

            OUT         (pandas.dataframe)      dataframe window of the current iteration
        '''

        start = self.nIteration - 1
        end = start + self.windowLength

        return self.data[start:end]

    def getSubjects(self):
        ''' returns a list of the available subjects of the trading data

        OUT             (list)      standardized names of the available subjects of each data symbol, e.g. ["price_ask_open", "macd_line"] '''

        return self.data.columns.to_list()

    def getPosition(self):
        ''' returns the position number of the dataset where the recent end position of the window is located
            example: after the 1st iteration and window length 5, the index of the windows last point is 4

        OUT             (int)      current index of the last windows position '''

        return (self.nIteration - 1) + (self.windowLength - 1)

    def getRemainingSteps(self):
        ''' returns the number of possible remaining iterations '''

        return (self.length - self.getPosition() - 1)

    def isLastWindow(self):
        ''' returns if the window that was returned by the last iteration is the last window in the dataset

        OUT             (bool)      if true, then window was the last window. no more iterations possible'''

        return self.getPosition() == self.length - 1

def smoothData(dataframe, averageSize):
    ''' smoothes the data in the dataframe according to a moving average and keeps the original data size

        IN  dataframe       (pandas.dataframe)      the dataframe with the data to smooth
        IN  averageSize     (int)                   the size of the rolling window
        OUT                 (pandas.dataframe)      smoothed dataframe
    '''

    return dataframe.rolling(averageSize, min_periods = 1 , center = True).mean()
