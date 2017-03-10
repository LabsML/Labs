from gzip import GzipFile

f = open('daal_retail.csv', 'w')
transactionId = 0;
for line in GzipFile('retail.dat.gz'):
    for item in line.strip().split():
        f.write(str(transactionId) + ',' + str(int(item)) + '\n')
    transactionId += 1
f.close()
