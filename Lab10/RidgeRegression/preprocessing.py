import pandas as pd
import csv

def writeToFile(filename, data, firstIndex, numberOfObjects):
    print('Start to write')
    csvfile = open(filename, 'w')
    csvfile.write('')
    for i in range(firstIndex, firstIndex + numberOfObjects):
        if i % 50000 == 0:
            csvfile.close()
            csvfile = open(filename, 'a')
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            print(i)
        row = data.iloc[i]
        writer.writerow(row)
    csvfile.close()

def drop_columns(dataSeries,threshold):
    for index in range(0,len(dataSeries)):
        if dataSeries[index] < threshold:
            return dataSeries[index:]
    return None

columnsName = ['Year','Month','DayofMonth','DayofWeek','CRSDepTime','CRSArrTime',
                'UniqueCarrier','FlightNum','ActualElapsedTime','Origin',
                'Dest','Distance','Diverted','ArrDelay']
data = pd.read_csv("1988_14col.data",names=columnsName)
print(data.shape)

cols = list(data.columns.values)
cols.reverse()
data=data[cols]

numberOfObjects = data.shape[0]
data.drop(['Year','DayofMonth','Month','FlightNum','Diverted'], axis = 1, inplace = True)
print(data.shape)

dataCategorial = None
categorial_cols = ['UniqueCarrier','DayofWeek', 'Origin', 'Dest']
for cc in categorial_cols:
    dummies = pd.get_dummies(data[cc], drop_first=False, sparse=True)
    dummies = dummies.add_prefix("{}#".format(cc))
    data.drop(cc, axis=1, inplace=True)
    if (dataCategorial is None):
        dataCategorial = dummies
    else:
        dataCategorial = dataCategorial.join(dummies)
dataCategorial.drop(['UniqueCarrier#PA (1)','DayofWeek#1','Dest#ABE','Origin#ABE'], axis = 1, inplace = True) #'Month#12'

cols = list(data.columns.values)
cols.reverse()
data=data[cols]
for column in data:
    print(column,end=';')
print()
data = data.to_sparse()
data = dataCategorial.join(data)
for column in data:
    print(column,end=';')
print()
print(data.shape)

numberOfTrainObjects = int(numberOfObjects * 0.8)
numberOfTestObjects = numberOfObjects - numberOfTrainObjects
writeToFile('1988_14col_train_data.csv',data,0,numberOfTrainObjects)
writeToFile('1988_14col_test_data.csv',data, numberOfTrainObjects, numberOfTestObjects)