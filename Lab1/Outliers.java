import com.intel.daal.algorithms.multivariate_outlier_detection.bacondense.InitializationMethod;
import com.intel.daal.algorithms.multivariate_outlier_detection.*;
import com.intel.daal.algorithms.multivariate_outlier_detection.Result;
import com.intel.daal.algorithms.multivariate_outlier_detection.bacondense.Batch;
import com.intel.daal.algorithms.multivariate_outlier_detection.bacondense.Parameter;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.services.DaalContext;

import java.nio.DoubleBuffer;

public class Outliers
{
    static
    {
        System.loadLibrary("JavaAPI");
    }

    private static final String datasetFileName = ""; //There should be the path to the file

    private static DaalContext context = new DaalContext();

    private static Result result;

    public static void main(String[] args)
    {
        FileDataSource dataSource = new FileDataSource(context, datasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        Batch algorithm = new Batch(context, Float.class, Method.baconDense);

        NumericTable input = dataSource.getNumericTable();
        algorithm.input.set(InputId.data,input);
        algorithm.parameter.setInitializationMethod(InitializationMethod.baconMedian);
        algorithm.parameter.setAlpha(0.05); //significance level parameter

        result = algorithm.compute();

        printMatrix();

        context.dispose();

    }

    public static void printMatrix() //function, which prints any Numeric Table
    {
        NumericTable corrmatrix = result.get(ResultId.weights);
        long r = corrmatrix.getNumberOfRows();
        long c = corrmatrix.getNumberOfColumns();
        System.out.println(r);
        System.out.println(c);
        DoubleBuffer buf = DoubleBuffer.allocate((int) (r*c));
        buf = corrmatrix.getBlockOfRows(0,r,buf);
        int pos = 0;
        for(int i = 0; i < r; i++)
        {
            for(int j = 0; j < c; j++)
            {
                if(buf.get(i) == 0)
                {
                    System.out.println("Object " + i);
                }
            }
        }
    }
}
