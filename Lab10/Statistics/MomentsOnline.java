import com.intel.daal.algorithms.classifier.training.*;
import com.intel.daal.algorithms.low_order_moments.*;
import com.intel.daal.algorithms.low_order_moments.InputId;
import com.intel.daal.algorithms.low_order_moments.PartialResultId;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.services.DaalContext;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.DoubleBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public class MomentsOnline
{
    private static final String dataset = ""; //There should be the path to the file

    private static final int    nVectorsInBlock = 10000; //Number of rows/observations in one block

    private static DaalContext context = new DaalContext();

    static PartialResult partialResult;

    static Result result;

    public static void main(String args[])
    {
        long fulltime = System.nanoTime();
        FileDataSource dataSource = new FileDataSource(context,dataset, DataSource.DictionaryCreationFlag.DoDictionaryFromContext, DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        Online algorithm = new Online(context, Double.class, Method.defaultDense);

        NumericTable data = dataSource.getNumericTable();
        algorithm.input.set(InputId.data, data);

        long time = System.nanoTime();
        while (dataSource.loadDataBlock(nVectorsInBlock) > 0)
        {
            partialResult = algorithm.compute();
        }

        result = algorithm.finalizeCompute();
        printMatrix(result.get(ResultId.mean));

        System.out.println("Time for online: " + (((float)(System.nanoTime() - time))/1000000000));
        System.out.println("Time for full online read and calculate: " + (((float)(System.nanoTime() - fulltime))/1000000000));
        context.dispose();
    }

    static void printMatrix(NumericTable table) //function, which prints any Numeric Table
    {
        long rows = table.getNumberOfRows();
        long cols = table.getNumberOfColumns();
        DoubleBuffer bufsup = DoubleBuffer.allocate((int)(rows*cols));
        bufsup = table.getBlockOfRows(0,rows,bufsup);
        if(cols == 1)
        {
            for(int i = 0; i < rows; i++)
            {
                System.out.print(" " + bufsup.get(i));
            }
            System.out.println();
            return;
        }
        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                System.out.print(" " + String.format("%.3f",bufsup.get(i * (int)cols + j)));
            }
            System.out.println();
        }
    }
}
