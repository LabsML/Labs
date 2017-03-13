import com.intel.daal.algorithms.logitboost.Model;
import com.intel.daal.algorithms.logitboost.prediction.PredictionBatch;
import com.intel.daal.algorithms.logitboost.prediction.PredictionMethod;
import com.intel.daal.algorithms.logitboost.training.TrainingBatch;
import com.intel.daal.algorithms.logitboost.training.TrainingMethod;
import com.intel.daal.algorithms.logitboost.training.TrainingResult;
import com.intel.daal.algorithms.classifier.prediction.ModelInputId;
import com.intel.daal.algorithms.classifier.prediction.NumericTableInputId;
import com.intel.daal.algorithms.classifier.prediction.PredictionResult;
import com.intel.daal.algorithms.classifier.prediction.PredictionResultId;
import com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix.BinaryConfusionMatrixInput;
import com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix.BinaryConfusionMatrixInputId;
import com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix.BinaryConfusionMatrixResult;
import com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix.BinaryConfusionMatrixResultId;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.svm.quality_metric_set.QualityMetricId;
import com.intel.daal.algorithms.svm.quality_metric_set.QualityMetricSetBatch;
import com.intel.daal.algorithms.svm.quality_metric_set.ResultCollection;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.MergedNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.services.DaalContext;

import java.nio.DoubleBuffer;

public class LogitBoost
{
    static { System.loadLibrary("JavaAPI"); }

    private static DaalContext context = new DaalContext();

    private static final String train = "";
    private static final String test = ""; //There should be paths to the files that contain train and test parts of dataset

    private static long nFeatures;

    private static TrainingResult trainingResult;
    private static PredictionResult predictionResult;
    private static NumericTable testGroundTruth;

    private static long maxIterations = 100;
    private static double accuracy = 0.01;

    public static void main(String args[])
    {
        trainModel();

        testModel();

        NumericTable predictedLabels = predictionResult.get(PredictionResultId.prediction);

        QualityMetricSetBatch metric = new QualityMetricSetBatch(context);
        BinaryConfusionMatrixInput input = metric.getInputDataCollection()
                .getInput(QualityMetricId.confusionMatrix);
        input.set(BinaryConfusionMatrixInputId.predictedLabels, predictedLabels);
        input.set(BinaryConfusionMatrixInputId.groundTruthLabels, testGroundTruth);

        ResultCollection quality = metric.compute();
        BinaryConfusionMatrixResult qualityMetricResult = quality.getResult(QualityMetricId.confusionMatrix);
        NumericTable table = qualityMetricResult.get(BinaryConfusionMatrixResultId.confusionMatrix);
        DoubleBuffer tab = DoubleBuffer.allocate(4);
        tab = table.getBlockOfRows(0,2,tab);
        System.out.println("" + tab.get(0) + " " + tab.get(1));
        System.out.println("" + tab.get(2) + " " + tab.get(3));


        NumericTable binaries = qualityMetricResult.get(BinaryConfusionMatrixResultId.binaryMetrics);
        DoubleBuffer buf = DoubleBuffer.allocate(6);
        buf = binaries.getBlockOfRows(0,1,buf);

        System.out.println("Accuracy: " + String.format("%.3f",buf.get(0)));
        /*System.out.println("Precision: " + buf.get(1));
        System.out.println("Recall: " + buf.get(2));
        System.out.println("fscore: " + buf.get(3));
        System.out.println("Specifity: " + buf.get(4));
        System.out.println("AUC: " + buf.get(5));*/

        context.dispose();
    }

    static void trainModel()
    {
        FileDataSource trainDataSource = new FileDataSource(context, train,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        nFeatures = trainDataSource.getNumberOfColumns();

        NumericTable trainData = new HomogenNumericTable(context, Double.class, nFeatures - 1, 0, NumericTable.AllocationFlag.NotAllocate);
        NumericTable trainGroundTruth = new HomogenNumericTable(context, Double.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(trainData);
        mergedData.addNumericTable(trainGroundTruth);

        trainDataSource.loadDataBlock(mergedData);

        TrainingBatch algorithm = new TrainingBatch(context, Double.class, TrainingMethod.friedman, 2);
        algorithm.input.set(InputId.data,trainData);
        algorithm.input.set(InputId.labels,trainGroundTruth);

        algorithm.parameter.setMaxIterations(maxIterations);
        algorithm.parameter.setAccuracyThreshold(accuracy);

        long time = System.nanoTime();
        trainingResult = algorithm.compute();
        System.out.println("Time to construct model: " + (System.nanoTime() - time));

    }

    static void testModel()
    {
        FileDataSource testDataSource = new FileDataSource(context, test,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        /* Create Numeric Tables for testing data and labels */
        NumericTable testData = new HomogenNumericTable(context, Double.class, nFeatures - 1, 0, NumericTable.AllocationFlag.NotAllocate);
        testGroundTruth = new HomogenNumericTable(context, Double.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(testGroundTruth);

        testDataSource.loadDataBlock(mergedData);

        PredictionBatch algorithm = new PredictionBatch(context, Double.class, PredictionMethod.defaultDense,2);

        Model model = trainingResult.get(TrainingResultId.model);
        algorithm.input.set(NumericTableInputId.data, testData);
        algorithm.input.set(ModelInputId.model, model);

        long time = System.nanoTime();
        predictionResult = algorithm.compute();
        System.out.println("Time to predict model: " + (System.nanoTime() - time));
    }

    static void printMatrix(NumericTable table)
    {
        long rows = table.getNumberOfRows();
        long cols = table.getNumberOfColumns();
        DoubleBuffer bufsup = DoubleBuffer.allocate((int)(rows*cols));
        bufsup = table.getBlockOfRows(0,rows,bufsup);
        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                System.out.print(" " + bufsup.get(i * (int)cols + j));
            }
            System.out.println();
        }
    }
}
