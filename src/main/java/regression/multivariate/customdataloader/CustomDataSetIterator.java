package regression.multivariate.customdataloader;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Multivariate Time Series Sequence Prediction using single layer LSTM
 *
 * In this example, we will show you how you can build your own iterator to preprocess data into batches of sequences before
 * training the model with the dataset.
 *
 *  1. Initialize the data
 *      We initialize the data int step 1.1 by creating an array of sequential data.
 *                {{10, 20, 30},
 *                 {20, 25, 45},
 *                 {30, 35, 65},
 *                 {40, 45, 85},
 *                 {50, 55, 105},
 *                 {60, 65, 125}};
 *
 *      But, the data need to be preprocessed into batches of sequence as shown below before training the data with the
 *      neural network.
 *      +-------+----------+-------+
 *      | .csv | Sequence | Label |
 *      +-------+----------+-------+
 *      | 0     | [10, 20] | 65    |
 *      |       | [20, 25] |       |
 *      |       | [30, 35] |       |
 *      +-------+----------+-------+
 *      | 1     | [20, 25] | 85    |
 *      |       | [30, 35] |       |
 *      |       | [40, 45] |       |
 *      +-------+----------+-------+
 *      | 2     | [30, 35] | 105   |
 *      |       | [40, 45] |       |
 *      |       | [50, 55] |       |
 *      +-------+----------+-------+
 *      | 3     | [40, 45] | 125   |
 *      |       | [50, 55] |       |
 *      |       | [60, 65] |       |
 *      +-------+----------+-------+
 *
 *      We can perform the pre-pocessing step by using the custom build iterator in step 1.2.
 *
 *  2. Setup the LSTM configuration
 *
 *  3. Setup UI server for training
 *
 *  4. Train the model
 *
 *  5. Perform time series predictions
 *
 * This example is inspired by Jason Brownlee from Machine Learning Mastery
 * Src: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
 *
 */
public class CustomDataSetIterator {

    private static int numFeatures = 2;
    private static double learningRate = 0.01;
    private static int epoch = 2000;

    public static void main(String[] args) {

         /*
		#### LAB STEP 1 #####
		1.1: Create sample data set
        */
        int[][] arr = new int[][]{
                {10, 20, 30},
                {20, 25, 45},
                {30, 35, 65},
                {40, 45, 85},
                {50, 55, 105},
                {60, 65, 125}};

        INDArray sequence = Nd4j.create(arr);

         /*
		#### LAB STEP 1 #####
		1.2: Preprocess data into [batch, numFeatures, timeSteps], see MyCustomTimeSeriesIterator for more details.
        */
        MyCustomTimeSeriesIterator trainIter = new MyCustomTimeSeriesIterator(sequence, 2, 3, 2, 2);

        //Optional: view data for each batch
        int j = 0;
        while (trainIter.hasNext()) {
            System.out.println("Batch: " + j);
            System.out.println(trainIter.next());
            j++;
        }

         /*
		#### LAB STEP 2 #####
		Build the model
        */
        int numInput = numFeatures;
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.NONE)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new LSTM.Builder()
                        .nIn(numInput)
                        .nOut(50)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new RnnOutputLayer.Builder()
                        .nIn(50)
                        .nOut(1)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        /*
		#### LAB STEP 3 #####
		Set listener
        */
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new StatsListener(storage, 10));

        /*
		#### LAB STEP 4 #####
		Train the model
        */
        model.fit(trainIter, epoch);

        /*
		#### LAB STEP 5 #####
		Inference
        */
        INDArray testInput1 = Nd4j.create(new double[][][]{{
                {10.0000, 20.0000, 30.0000},
                {20.0000, 25.0000, 35.0000}}});
        System.out.println(model.output(testInput1));

        INDArray testInput2 = Nd4j.create(new double[][][]{{
                {20.0000, 30.0000, 40.0000},
                {25.0000, 35.0000, 45.0000}}});
        System.out.println(model.output(testInput2));

        INDArray testInput3 = Nd4j.create(new double[][][]{{
                {30.0000, 40.0000, 50.0000},
                {35.0000, 45.0000, 55.0000}}});
        System.out.println(model.output(testInput3));

    }
}
