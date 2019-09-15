import os
import csv
import glob
import zipfile
import pandas as pd
from datetime import datetime
from tqdm import tqdm, trange


class loader:
    def __init__(self):
        self.KLOTPath = "/media/gh05t/New Volume/dataset/air polution/Meteorologicaldata/Meteorological_data/KLOT/"
        self.KIGQPath = "/media/gh05t/New Volume/dataset/air polution/Meteorologicaldata/Meteorological_data/KIGQ/"
        self.pollution = "/media/gh05t/New Volume/dataset/air polution/daily/ozone/"
        self.metroCSV = []
        self.pollutionCSV = None

    def metroLoader(self):
        data = []
        # df = pd.DataFrame()
        for filename in tqdm(glob.glob(self.KLOTPath + "*.csv")):
            cv = list(csv.reader(open(filename, "r"), delimiter=","))
            for line in range(8, len(cv)):
                self.metroCSV.append(cv[line])
                # data.append(pd.read_csv(filename, sep=',', skiprows=8, error_bad_lines=False, sort=True))
        for filename in tqdm(glob.glob(self.KIGQPath + "*.csv")):
            cv = list(csv.reader(open(filename, "r"), delimiter=","))
            for line in range(8, len(cv)):
                self.metroCSV.append(cv[line])
        self.metroCSV.sort()
        pd.DataFrame(self.metroCSV).to_csv("file.csv")

    def pollutionLoader(self):
        data = []
        for filename in tqdm(glob.glob(self.pollution + "*.zip")):
            zf = zipfile.ZipFile((filename))
            df = pd.DataFrame(pd.read_csv(zf.open(filename[-21:-4] + ".csv"), delimiter=",", encoding='utf-8-sig'))
            is_cook = df["County Name"] == "Cook"
            df['Date Local'] = pd.to_datetime(df['Date Local'].astype(str), format='%Y/%m/%d')
            df['time'] = df["Date Local"].dt.strftime('%m/%d/%Y').map(str) + ' ' + df["Time Local"]
            df = df[is_cook].groupby(['time'])['Sample Measurement'].mean()
            
            data.append(df)
        self.pollutionCSV = pd.concat(data)

        pd.DataFrame(self.pollutionCSV).to_csv("so2.csv")

    def loadAll(self):
        self.KLOTLoader()
        self.metroLoader()
        self.pollutionLoader()
