import pandas as pd
import csv

def writeToFileDataframe(filename, data):
    print('Start to write into ' + filename)
    csvfile = open(filename, 'w')
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    for index, row in data.iterrows():
        writer.writerow(row)
    csvfile.close()
    print('Finish to write')

def writeToFileSeries(filename, data):
    print('Start to write into ' + filename)
    csvfile = open(filename, 'w')
    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
    for index in range(0,len(data)):
        row = [data.iloc[index]]
        writer.writerow(row)
    csvfile.close()
    print('Finish to write')

def writeToFiles(filename, extension, data, firstIndex, numberOfObjects, maxObjectsInFile):
    fileNumber = 1
    while (numberOfObjects > 0):
        newFilename = filename + '_' + str(fileNumber) + extension
        newNumberOfObjects = maxObjectsInFile
        if numberOfObjects < maxObjectsInFile:
            newNumberOfObjects = numberOfObjects
        newData = data[firstIndex:firstIndex+newNumberOfObjects]
        if (type(newData) is pd.Series):
            writeToFileSeries(newFilename, newData)
        else:
            writeToFileDataframe(newFilename, newData)
        firstIndex += newNumberOfObjects
        numberOfObjects -= newNumberOfObjects
        fileNumber += 1

def drop_columns(dataSeries,threshold):
    for index in range(0,len(dataSeries)):
        if dataSeries[index] < threshold:
            return dataSeries[index:]
    return None

columnsName = ['Year','Month','DayofMonth','DayofWeek','CRSDepTime','CRSArrTime',
                'UniqueCarrier','FlightNum','ActualElapsedTime','Origin',
                'Dest','Distance','Diverted','ArrDelay']
data = pd.read_csv("airline_14col.data",names=columnsName)
print(data.shape)
data = data[data['Year'] <= 1991]
print(data.shape)
for column in data:
    print(column,end=';')
print()
numberOfObjects = data.shape[0]
data.drop(['Year','DayofMonth','FlightNum','Diverted'], axis = 1, inplace = True)
print(data.shape)

dataCategorial = None
categorial_cols = ['UniqueCarrier','Month','DayofWeek', 'Origin', 'Dest']
for cc in categorial_cols:
    dummies = pd.get_dummies(data[cc], drop_first=False, sparse=True)
    dummies = dummies.add_prefix("{}#".format(cc))
    data.drop(cc, axis=1, inplace=True)
    if (dataCategorial is None):
        dataCategorial = dummies
    else:
        dataCategorial = dataCategorial.join(dummies)
    print(dataCategorial.shape)
data = data.to_sparse()
data = dataCategorial.join(data)
print(data.shape)
for column in data:
    print(column,end=';')
print()
numberOfTrainObjects = int(numberOfObjects * 0.8)
numberOfTestObjects = numberOfObjects - numberOfTrainObjects
print(numberOfTrainObjects)
print(numberOfTestObjects)

maxObjectsInFile = 50000
writeToFiles('train_features', '.csv', data, 0, numberOfTrainObjects, maxObjectsInFile)
writeToFiles('test_features', '.csv', data, numberOfTrainObjects, numberOfTestObjects, maxObjectsInFile)