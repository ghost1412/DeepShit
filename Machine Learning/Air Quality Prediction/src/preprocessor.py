from loader import *
from utils import *
import nltk
import numpy as np
import calendar
from tqdm import tqdm, trange


class preprocessor:
    def __init__(self):
        self.path = None
        self.data = loader()
        self.processedmetro = None
        self.windir = None
        self.weather = None
        self.meanmetro = None
        self.helper = utils()

    def createMean(self, data):
        newData = []
        row = 0
        numSet = [3, 5, 6, 7, 8, 11, 17, 25, 28]
        wind = [0] * 16
        pbar = tqdm(total=len(data))
        while row <= len(data)- 1:
            try:
                colNo = 1
                meanRow = [[]] * 10
                meanData = []
                for line in range(row, len(data) - 1):
                    meanData.append(data[row])
                    if self.helper.ntSameDataTime(
                        data[row][1][0:13], data[row + 1][1][0:13]
                    ):
                        break
                    row += 1
                    pbar.update(1)
                meanData = np.reshape(meanData, (len(meanData), 33))
                meanData[meanData == np.nan] = 0
                meanRow[0] = meanData[0][1][:-6]+'00'
                for col in numSet:
                    meanRow[colNo] = np.mean(meanData[:, col].astype(np.float))
                    colNo += 1
                meanRow += self.createCat(
                    "wind", nltk.FreqDist(meanData[:, 27]).max()
                ) + self.createCat("weather", nltk.FreqDist(meanData[:, -2]).max())
                row += 1
                newData.append(meanRow)
                pbar.update(1)
            except:
                row += 1
                pbar.update(1)
                # print(newData)
        pd.DataFrame(newData).to_csv("metro.csv")
        return newData

    def interpolate(self, data):
        df = pd.DataFrame(data)
        df = df.replace("", np.nan, regex=True)
        # for col in range(0, 32):
        df = df.interpolate().ffill().bfill()
        """for col in range(0, 24):
			try:
				df[:, col] = df[:, col].interpolate(method='linear', limit_direction='forward', axis=0)
			except: 
				continue"""
        return np.array(df)

    def createCat(self, dtype, val):
        if dtype == "wind":
            hotWind = [0] * len(self.windir)
            hotWind[self.windir.index(val)] = 1
            return hotWind
        elif dtype == "weather":
            hotWeather = [0] * len(self.weather)
            hotWeather[self.weather.index(val)] = 1
            return hotWeather

    def weekType(self, data):
        df = df.DataFrame(data)
        df['weekType'] = calender.weekday(df["time"].dt.strftime('%m/%d/%Y').map(str))
        print(df)

    def metroProvider(self):
        self.data.metroLoader()
        data = self.interpolate(self.data.metroCSV)
        self.windir = self.helper.windSet(data)
        self.weather = self.helper.weatherSet(data)

        metro = pd.DataFrame(self.createMean(data))
        return metro

    def processData(self):

        metro = (self.metroProvider())
        colist = ['s' + str(x) for x in range(0,52)]
        colist[0] = 'time'

        print(colist)
        metro.columns = colist
        self.data.pollutionLoader()
        gasData = self.data.pollutionCSV
        finalData = pd.merge(gasData, metro, on="time")
        pd.DataFrame(finalData).to_csv("gasname.csv")
        return finalData


# if __name__ == "__main__":
#a = preprocessor()
#a.processData()
