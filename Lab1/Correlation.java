import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.covariance.*;

import java.nio.DoubleBuffer;

public class Correlation
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
        dataSource.loadDataBlock(50); //loadDataBlock() loads whole file. loadDataBlock(50) loads first 50 rows
        Batch algorithm = new Batch(context, Double.class, Method.defaultDense);

        NumericTable input = dataSource.getNumericTable();
        algorithm.input.set(InputId.data,input);

        algorithm.parameter.setOutputMatrixType(OutputMatrixType.correlationMatrix);

        result = algorithm.compute();

        printMatrix();

        context.dispose();


    }

    public static void printMatrix() //function, which prints a Numeric Table
    {
        NumericTable corrmatrix = result.get(ResultId.correlation);
        long r = corrmatrix.getNumberOfRows();
        long c = corrmatrix.getNumberOfColumns();
        DoubleBuffer buf = DoubleBuffer.allocate((int) (r*c));
        buf = corrmatrix.getBlockOfRows(0,r,buf);
        int pos = 0;
        for(int i = 0; i < r; i++)
        {
            for(int j = 0; j < c; j++)
            {
                System.out.print(buf.get(pos++));
                System.out.print("  ");
            }
            System.out.println();
        }
    }
}
