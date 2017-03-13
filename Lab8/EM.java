import com.intel.daal.algorithms.em_gmm.*;
import com.intel.daal.algorithms.em_gmm.init.*;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.services.DaalContext;

import java.nio.DoubleBuffer;

public class EM
{
    static { System.loadLibrary("JavaAPI"); }

    private static DaalContext context = new DaalContext();

    private static final String dataset = ""; //There should be the path to the file

    private static long nComponents = 50; //Number of clusters/gaussian components

    public static void main(String args[])
    {
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        NumericTable input = dataSource.getNumericTable();

        InitBatch init = new InitBatch(context, Double.class, InitMethod.defaultDense,nComponents);
        init.input.set(InitInputId.data, input);
        init.parameter.setNDepthIterations(1000);
        init.parameter.setNTrials(1);
        InitResult initresult = init.compute();

        Batch algorithm = new Batch(context, Double.class, Method.defaultDense,nComponents);
        algorithm.input.set(InputId.data,input);
        algorithm.input.set(InputValuesId.inputValues, initresult);

        long time = System.nanoTime();
        Result result = null;
        for(int i = 0; i < 10; i++)
        {
            initresult = init.compute();
            algorithm.input.set(InputValuesId.inputValues, initresult);

            result = algorithm.compute();
            printMatrix(result.get(ResultId.goalFunction));
        }
        System.out.println(System.nanoTime() - time);


        printMatrix(result.get(ResultId.goalFunction));
        printMatrix(result.get(ResultId.means));
        printMatrix(result.get(ResultId.weights));


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
            System.out.println();
            return;
        }
        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                System.out.print(bufsup.get(i * (int)cols + j) + " ");
            }
            System.out.println();
        }
        System.out.println();
    }
}
