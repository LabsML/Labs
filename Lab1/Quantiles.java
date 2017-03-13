import com.intel.daal.algorithms.quantiles.*;
import com.intel.daal.algorithms.quantiles.Batch;
import com.intel.daal.algorithms.quantiles.Method;
import com.intel.daal.algorithms.quantiles.Result;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.services.DaalContext;

import java.nio.DoubleBuffer;

public class Quantiles
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

        Batch algorithm = new Batch(context, Double.class, Method.defaultDense);

        NumericTable input = dataSource.getNumericTable();
        algorithm.input.set(InputId.data, input);

        double data[] = {0.25, 0.5, 0.75}; //vector of quantiles
        HomogenNumericTable quantileOrders = new HomogenNumericTable(context, data, 3, 1);
        algorithm.parameter.setQuantileOrders(quantileOrders);
        result = algorithm.compute();

        NumericTable table = result.get(ResultId.quantiles);
        long r = table.getNumberOfRows();
        long c = table.getNumberOfColumns();
        DoubleBuffer buf = DoubleBuffer.allocate((int) (r * c));
        buf = table.getBlockOfRows(0, r, buf);

        for(int i = 0; i < r*c; i++)
        {
            if(i % c == 0)
                System.out.println();
            System.out.println(buf.get(i));
        }

        context.dispose();
    }
}