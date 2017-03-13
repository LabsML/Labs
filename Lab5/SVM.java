import com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix.BinaryConfusionMatrixInput;
import com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix.BinaryConfusionMatrixInputId;
import com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix.BinaryConfusionMatrixResult;
import com.intel.daal.algorithms.classifier.quality_metric.binary_confusion_matrix.BinaryConfusionMatrixResultId;

import com.intel.daal.algorithms.svm.quality_metric_set.QualityMetricId;
import com.intel.daal.algorithms.svm.quality_metric_set.QualityMetricSetBatch;
import com.intel.daal.algorithms.svm.quality_metric_set.ResultCollection;
import com.intel.daal.data_management.data.*;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.classifier.prediction.*;
import com.intel.daal.algorithms.classifier.training.InputId;
import com.intel.daal.algorithms.classifier.training.TrainingResultId;
import com.intel.daal.algorithms.svm.Model;
import com.intel.daal.algorithms.svm.prediction.PredictionBatch;
import com.intel.daal.algorithms.svm.prediction.PredictionMethod;
import com.intel.daal.algorithms.svm.training.TrainingBatch;
import com.intel.daal.algorithms.svm.training.TrainingMethod;
import com.intel.daal.algorithms.svm.training.TrainingResult;

import java.nio.DoubleBuffer;

public class SVM
{
    static { System.loadLibrary("JavaAPI");}

    private static DaalContext context = new DaalContext();

    private static TrainingResult training;
    private static PredictionResult testing;
    private static NumericTable testGroundTruth;

    private static ResultCollection quality;

    private static final String trainDatasetFileName     = "";
    private static final String testDatasetFileName     = ""; //There should be paths to files with train and test parts of dataset

    private static long nFeatures = 20; //The number of columns, including labels for classes. Labels should be -1 and 1.

    private static com.intel.daal.algorithms.kernel_function.rbf.Batch rbfKernel;
    private static com.intel.daal.algorithms.kernel_function.linear.Batch linearKernel;

    public static void main(String[] args)
    {
        linearKernel = new com.intel.daal.algorithms.kernel_function.linear.Batch(context, Double.class);
        rbfKernel = new com.intel.daal.algorithms.kernel_function.rbf.Batch(context, Double.class);

        System.out.println("Train Model");
        trainModel();
        System.out.println("Test Model");
        testModel();

        NumericTable predictedLabels = testing.get(PredictionResultId.prediction);

        Model model = training.get(TrainingResultId.model);

        QualityMetricSetBatch metric = new QualityMetricSetBatch(context);
        BinaryConfusionMatrixInput input = metric.getInputDataCollection()
                .getInput(QualityMetricId.confusionMatrix);
        input.set(BinaryConfusionMatrixInputId.predictedLabels, predictedLabels);
        input.set(BinaryConfusionMatrixInputId.groundTruthLabels, testGroundTruth);

        quality = metric.compute();
        BinaryConfusionMatrixResult qualityMetricResult = quality.getResult(QualityMetricId.confusionMatrix);
        NumericTable table = qualityMetricResult.get(BinaryConfusionMatrixResultId.confusionMatrix);
        DoubleBuffer tab = DoubleBuffer.allocate(4);
        tab = table.getBlockOfRows(0,2,tab);
        //Printing confusion matrix
        System.out.println("" + tab.get(0) + " " + tab.get(1));
        System.out.println("" + tab.get(2) + " " + tab.get(3));

        NumericTable binaries = qualityMetricResult.get(BinaryConfusionMatrixResultId.binaryMetrics);
        DoubleBuffer buf = DoubleBuffer.allocate(6);
        buf = binaries.getBlockOfRows(0,1,buf);


        //Different metrics of confusion matrix
        System.out.println("Accuracy: " + buf.get(0));
        System.out.println("Precision: " + buf.get(1));
        System.out.println("Recall: " + buf.get(2));
        System.out.println("fscore: " + buf.get(3));
        System.out.println("Specifity: " + buf.get(4));
        System.out.println("AUC: " + buf.get(5));

        context.dispose();
    }

    private static void trainModel()
    {
        FileDataSource trainDataSource = new FileDataSource(context, trainDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);
        nFeatures = trainDataSource.getNumberOfColumns();
        System.out.println(nFeatures);

        NumericTable trainData = new HomogenNumericTable(context, Double.class,nFeatures - 1, 0, NumericTable.AllocationFlag.NotAllocate);
        NumericTable trainGroundTruth = new HomogenNumericTable(context, Double.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(trainData);
        mergedData.addNumericTable(trainGroundTruth);
        System.out.println("Load");
        trainDataSource.loadDataBlock(mergedData);
        System.out.println("Batch");
        TrainingBatch algorithm = new TrainingBatch(context, Double.class, TrainingMethod.boser);
        System.out.println("Kernel");
        algorithm.parameter.setKernel(rbfKernel);
        algorithm.parameter.setMaxIterations(100000);
        algorithm.parameter.setCacheSize(200000000); //Cashe size is usually > n^2 * sizeof(double) where n is the number of rows/observations
        /* Pass a training data set and dependent values to the algorithm */
        algorithm.input.set(InputId.data, trainData);
        algorithm.input.set(InputId.labels, trainGroundTruth);
        System.out.println("Compute");

        long time = System.nanoTime();
        training = algorithm.compute();
        System.out.println("The time for calculating the model is " + (System.nanoTime() - time));
    }

    private static void testModel()
    {
        FileDataSource testDataSource = new FileDataSource(context, testDatasetFileName,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.NotAllocateNumericTable);

        NumericTable testData = new HomogenNumericTable(context, Double.class, nFeatures - 1, 0, NumericTable.AllocationFlag.NotAllocate);
        testGroundTruth = new HomogenNumericTable(context, Double.class, 1, 0, NumericTable.AllocationFlag.NotAllocate);
        MergedNumericTable mergedData = new MergedNumericTable(context);
        mergedData.addNumericTable(testData);
        mergedData.addNumericTable(testGroundTruth);

        testDataSource.loadDataBlock(mergedData);

        PredictionBatch algorithm = new PredictionBatch(context, Double.class, PredictionMethod.defaultDense);


        algorithm.parameter.setKernel(rbfKernel);

        Model model = training.get(TrainingResultId.model);

        algorithm.input.set(NumericTableInputId.data, testData);
        algorithm.input.set(ModelInputId.model, model);

        long time = System.nanoTime();
        testing = algorithm.compute();
        System.out.println("The time for calculating the prediction is " + (System.nanoTime() - time));
    }


    static void printMatrix(NumericTable table) //function, which prints a Numeric Table
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


