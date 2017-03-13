import com.intel.daal.algorithms.low_order_moments.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.services.DaalContext;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;

class LowOrderMoments
{
    static { System.loadLibrary("JavaAPI"); }

    private static final String dataset = ""; //There should be the path to the file

    private static DaalContext context = new DaalContext();

    private static Result result;

    static DoubleBuffer getData(ResultId id) //Function for obtaining buffer with data
    {
        NumericTable table = result.get(id);
        long r = table.getNumberOfRows();
        long c = table.getNumberOfColumns();
        DoubleBuffer buf = DoubleBuffer.allocate((int) (r * c));
        buf = table.getBlockOfRows(0, r, buf);
        return buf;
    }

    public static void main(String[] args)
    {
        long fulltime = System.nanoTime();
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        dataSource.loadDataBlock();

        Batch algorithm = new Batch(context, Double.class, Method.defaultDense);

        NumericTable input = dataSource.getNumericTable();
        algorithm.input.set(InputId.data, input);

        System.out.println("Ready to compute");
        long time = System.nanoTime();
        result = algorithm.compute();
        System.out.println("Total computing time: " + (System.nanoTime() - time));
        System.out.println("Total computing and reading time: " + (System.nanoTime() - fulltime));
        DoubleBuffer buf;

        buf = getData(ResultId.sum);
        System.out.println("Sum: "+ buf.get(0)); //buf.get(0) obtains the correspondent statistic for first column
        buf = getData(ResultId.mean);
        System.out.println("Mean: "+ buf.get(0));
        buf = getData(ResultId.standardDeviation);
        System.out.println("Standard deviation: "+ buf.get(0));
        buf = getData(ResultId.minimum);
        System.out.println("Minimum: "+ buf.get(0));
        buf = getData(ResultId.maximum);
        System.out.println("Maximum: "+ buf.get(0));

        printMatrix(result.get(ResultId.mean));

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

