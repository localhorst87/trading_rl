import pickle
import random
import numpy as np
import os.path
import pandas as pd
from Dataset import *
from datetime import datetime, timedelta

class Sampler:
    '''
    The Sampler class is used for returning randomized snippets of financial data.

    The getRandomDataset method returns data of 1000 samples of a random data symbol for a random date
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

    def getRandomDataset(self, windowLength, sampleLength):
        ''' creates a dataset of the loaded data of random date

        IN      windowLength    (integer)   defines the length of the window the dataset should return each iteration step
        OUT                     (Dataset)   a random Dataset of with the given parameters '''

        if windowLength >= sampleLength: raise ValueError("window length must be smaller than the sampe length")

        dataComplete = False

        while not dataComplete:
            forexSymbol = self._getRandomTradingSymbol()
            datetimeEnd = self._getRandomDatetime()
            dataframeCut = self._cutDataframe(self.data[forexSymbol], datetimeEnd, sampleLength) # cuts data by length
            if len(dataframeCut) < sampleLength or dataframeCut.isnull().values.any(): continue # price data not complete, try another sample

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
