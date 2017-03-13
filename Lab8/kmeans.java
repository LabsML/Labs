import com.intel.daal.algorithms.em_gmm.init.*;
import com.intel.daal.algorithms.kmeans.init.*;
import com.intel.daal.algorithms.kmeans.init.InitBatch;
import com.intel.daal.algorithms.kmeans.init.InitInputId;
import com.intel.daal.algorithms.kmeans.init.InitMethod;
import com.intel.daal.algorithms.kmeans.init.InitResult;
import com.intel.daal.algorithms.kmeans.init.InitResultId;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.kmeans.*;

import java.nio.DoubleBuffer;

public class Cluster
{
    static { System.loadLibrary("JavaAPI"); }

    private static DaalContext context = new DaalContext();

    private static final String dataset = ""; //There should be the path to the file

    private static final int nClusters = 5; //number of desired clusters

    private static final int maxIterations = 10000;

    public static void main(String[] args)
    {
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        NumericTable input = dataSource.getNumericTable();

        InitBatch init = new InitBatch(context, Double.class, InitMethod.randomDense, nClusters);
        init.input.set(InitInputId.data, input);

        InitResult initResult = init.compute();
        NumericTable inputCentroids = initResult.get(InitResultId.centroids);

        Batch algorithm = new Batch(context, Double.class, Method.lloydDense, nClusters, maxIterations);

        algorithm.input.set(InputId.data, input);
        algorithm.input.set(InputId.inputCentroids, inputCentroids);
        algorithm.parameter.setMaxIterations(maxIterations);

        long time = System.nanoTime();
        Result result = null;
        for(int i = 0; i < 10; i++)
        {
            init.compute();
            inputCentroids = initResult.get(InitResultId.centroids);
            algorithm.input.set(InputId.inputCentroids, inputCentroids);
            result = algorithm.compute();
            printMatrix(result.get(ResultId.goalFunction));
            System.out.println();
        }
        System.out.println(System.nanoTime() - time);

        printMatrix(result.get(ResultId.goalFunction));
        System.out.println();
        printMatrix(result.get(ResultId.centroids));
        System.out.println();
        assignnments(result.get(ResultId.assignments));
        System.out.println();
        printMatrix(result.get(ResultId.assignments));


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
                System.out.print(" " + bufsup.get(i * (int)cols + j));
            }
            System.out.println();
        }
    }

    static void assignnments(NumericTable assignment) //special assignments printing function
    {
        long rows = assignment.getNumberOfRows();
        long cols = assignment.getNumberOfColumns();
        DoubleBuffer bufsup = DoubleBuffer.allocate((int)(rows*cols));
        bufsup = assignment.getBlockOfRows(0,rows,bufsup);
        for(int i = 0; i < nClusters; i++)
        {
            int count = 0;
            for(int j = 0; j < rows; j++)
            {
                if(bufsup.get(j) == i)
                    count++;
            }
            System.out.print(" " + count);
        }
    }
}