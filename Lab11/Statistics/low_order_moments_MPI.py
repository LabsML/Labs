import sys
from os import environ
from os.path import join as jp

from mpi4py import MPI

from daal import step1Local, step2Master
import daal.algorithms.low_order_moments as low_order_moments

from daal.data_management import OutputDataArchive, InputDataArchive, FileDataSource, DataSourceIface, BlockDescriptor, readOnly

import time

nBlocks = 4 #number of processes for one node

MPI_ROOT = 0 #id of master-process

DATA_PREFIX = "" #The name of the folder with dataset files

datasetfilenames = [
    jp(DATA_PREFIX,'low1.csv'),
    jp(DATA_PREFIX,'low2.csv'),
    jp(DATA_PREFIX,'low3.csv'),
    jp(DATA_PREFIX,'low4.csv')
] #Names of datafiles for every process

if __name__ == "__main__":

    timed_total = time.process_time()
    comm = MPI.COMM_WORLD
    rankId = comm.Get_rank()


    dataSource = FileDataSource(datasetfilenames[rankId],
                                DataSourceIface.doAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext) #Every process reads its own file

    dataSource.loadDataBlock()
    timed = time.process_time()
    localAlgorithm = low_order_moments.Distributed(step=step1Local)
    localAlgorithm.input.set(low_order_moments.data, dataSource.getNumericTable())
    pres = localAlgorithm.compute()
    #serializing results for sending to master-node
    dataArch = InputDataArchive()
    pres.serialize(dataArch)

    nodeResults = dataArch.getArchiveAsArray()
    serializedData = comm.gather(nodeResults)
    if rankId == MPI_ROOT:
        masterAlgorithm = low_order_moments.Distributed(step=step2Master)

        for i in range(nBlocks):
            #reading serialized data from other nodes
            dataArch = OutputDataArchive(serializedData[i])

            dataForStep2FromStep1 = low_order_moments.PartialResult()
            dataForStep2FromStep1.deserialize(dataArch)
            masterAlgorithm.input.add(low_order_moments.partialResults, dataForStep2FromStep1)

        print(time.process_time() - timed_total)
        print(time.process_time() - timed)

        masterAlgorithm.compute()
        res = masterAlgorithm.finalizeCompute()

        data_table = res.get(low_order_moments.mean)
        block = BlockDescriptor()
        rows = data_table.getNumberOfRows()
        data_table.getBlockOfRows(0, rows, readOnly, block)
        arr = block.getArray()
        print(arr) #print the results of computation