package exercise.stockprediction;

import exercise.stockprediction.utils.Plot;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Stock price prediction using LSTM
 *
 * Steps:
 *  1. Understand your data
 *  2. Create a Schema
 *  3. Create TransformProcess
 *  4. Load in the data via RecordReader and perform transformation using TransformProcess
 *  5. Separate the data into batches via iterator
 *  6. Model creation:
 *      6.1 create model configuration
 *      6.2 build the model based on the configuration
 *  7. Set Listener
 *  8. Train the model
 *  9. Evaluate the model
 *  10.Visualize all test labels vs predictions
 */
public class Solution {

    private static final double learningRate = 0.001;
    private static INDArray processedDataArray;
    private static int numFeatures = 4;
    private static int numLinesToSkip = 1;
    private static char delimiter = ',';

    public static void main(String[] args) throws Exception {
        Nd4j.getEnvironment().allowHelpers(false);

        /*
		#### LAB STEP 2 #####
		Create a schema
        */
        Schema schema = getSchema();

        /*
		#### LAB STEP 3 #####
		Create a TransformProcess
        */
        TransformProcess tp = getTransformProcess(schema);

        /*
		#### LAB STEP 4 #####
		Load in the data via RecordReader and perform transformation using TransformProcess
        */
        File data = new ClassPathResource("/datasets/StockData/AAPL_stockprice.csv").getFile();
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(data));

        List<List<Writable>> originalData = new ArrayList<>();
        while (recordReader.hasNext()) {
            originalData.add(recordReader.next());
        }
        List<List<Writable>> transformedData = LocalTransformExecutor.execute(originalData, tp);
        processedDataArray = RecordConverter.toMatrix(transformedData);

        List<INDArray> trainTestList = trainTestSplit(processedDataArray, 0.80);
        INDArray trainingSet = trainTestList.get(0);
        INDArray testSet = trainTestList.get(1);

         /*
		#### LAB STEP 5 #####
		Separate the data into batches via iterator. (See StockIterator for more details)
        */
        StockIterator trainIter = new StockIterator(trainingSet, 10, 8, numFeatures, 4);
        StockIterator testIter = new StockIterator(testSet, 1, 8, numFeatures, 4);

        //uncomment the following code. Can you tell what is the number of time step used to predict the next data?
        //System.out.println(trainIter.next());

        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler();
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);
        trainIter.reset();

        MultiLayerConfiguration config = getMultiLayerConfiguration();

        /*
		#### LAB STEP 7 #####
		Set listener
        */
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new StatsListener(storage, 10));

        /*
		#### LAB STEP 8 #####
		Train the model
        */
        for (int i = 0; i < 3; i++) {
            System.out.println("Epoch: " + i);
            while (trainIter.hasNext()) {
                DataSet ds = trainIter.next();
                model.fit(ds);
            }
            trainIter.reset();
            RegressionEvaluation re = model.evaluateRegression(testIter);
            System.out.println(re.stats());
            testIter.reset();
        }

        /*
		#### LAB STEP 9 #####
		Evaluate the model
        */

        long numBatches = testIter.getNumBatches();
        INDArray labelsForPlotting = Nd4j.zeros(numBatches);
        INDArray predictionsForPlotting = Nd4j.zeros(numBatches);

        int i = 0;
        while (testIter.hasNext()) {
            DataSet testDataRow = testIter.next();
            INDArray featureSequence = testDataRow.getFeatures();
            INDArray labels = testDataRow.getLabels();
            INDArray predictions = model.output(featureSequence);
            //take the value of last time step (Sequence Length -1)
            INDArray labelLastTimeStep = labels.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(7)});
            INDArray predictionLastTimeStep = predictions.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(7)});
            //optional
            //printData(featureSequence, labelLastTimeStep, predictionLastTimeStep);
            labelsForPlotting.put(i, labelLastTimeStep);
            predictionsForPlotting.put(i, predictionLastTimeStep);
            i++;
        }

        /*
		#### LAB STEP 10 #####
		Visualize all test labels vs predictions
        */
        plotData(labelsForPlotting, "Labels");
        plotData(predictionsForPlotting, "Predictions");

    }

    private static MultiLayerConfiguration getMultiLayerConfiguration() {
        return new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(learningRate))
                    .list()
                    .layer(0, new LSTM.Builder()
                            .nIn(numFeatures)
                            .nOut(100)
                            .activation(Activation.TANH)
                            .build())
                    .layer(1, new RnnOutputLayer.Builder()
                            .nIn(100)
                            .nOut(1)
                            .lossFunction(LossFunctions.LossFunction.MSE)
                            .activation(Activation.IDENTITY)
                            .build())
                    .build();
    }

    private static TransformProcess getTransformProcess(Schema schema) {
        return new TransformProcess.Builder(schema)
                    .removeColumns("Date")
                    .removeColumns("Volume")
                    .build();
    }

    private static Schema getSchema() {
        return new Schema.Builder()
                    .addColumnString("Date")
                    .addColumnDouble("High")
                    .addColumnDouble("Low")
                    .addColumnDouble("Close")
                    .addColumnDouble("Adj Close")
                    .addColumnLong("Volume")
                    .addColumnDouble("Open")
                    .build();
    }

    private static void printData(INDArray featureSequence, INDArray labelLastTimeStep, INDArray predictionLastTimeStep) {
        System.out.println("test feature: \n" + featureSequence);
        System.out.println("test label: \n" + labelLastTimeStep);
        System.out.println("prediction: \n" + predictionLastTimeStep);
    }

    private static List<INDArray> trainTestSplit(INDArray input, double trainPerc) {

        long sampleSize = input.shape()[0];
        int trainSize = (int) Math.floor(sampleSize * trainPerc);

        INDArray trainArray = input.get(NDArrayIndex.interval(0, trainSize), NDArrayIndex.all());
        INDArray testArray = input.get(NDArrayIndex.interval(trainSize, sampleSize), NDArrayIndex.all());

        return Arrays.asList(trainArray, testArray);
    }

    private static void plotData(INDArray labels, String title) {
        int numDays = labels.columns();
        INDArray days = Nd4j.arange(0, numDays); //as index
        Plot plot = new Plot(days, labels, title);
        plot.display();
    }

}
