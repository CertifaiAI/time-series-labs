package solution.regression.Basic.Univariate;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class UnivariateBidirectionalLSTM {

    private static double learningRate = 0.001;

    public static void main(String[] args) {
        // Step 1: Initialize the
        double[] sequenceData = new double[]{10, 20, 30, 40, 50, 60, 70, 80, 90};
        TimeSeriesUnivariateData data = new TimeSeriesUnivariateData(sequenceData,3,1);
        INDArray feature = data.getFeatureMatrix();
        INDArray label = data.getLabels();
        int sequenceLength = data.getSequenceLength();

        System.out.println("feature size: " + feature.shapeInfoToString());
        System.out.println("label size: " + label.shapeInfoToString());

        System.out.println(feature);
        System.out.println(label);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(learningRate))
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .miniBatch(false)
                .list()
                .layer(0, new Bidirectional(new LSTM.Builder()
                        .nIn(sequenceLength)
                        .nOut(50)
                        .activation(Activation.TANH)
                        .build()))
                .layer(1, new RnnOutputLayer.Builder()
                        .nIn(100)
                        .nOut(1)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.init();

        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);
        network.setListeners(new StatsListener(storage, 10));
        for (int i = 0; i < 10000; i++) {
            network.fit(feature, label);
        }

        INDArray testInput1 = Nd4j.create(new double[]{10, 20, 30}).reshape(new int[]{1, 3, 1});
        System.out.println(network.output(testInput1));

        INDArray testInput2 = Nd4j.create(new double[]{20, 30, 40}).reshape(new int[]{1, 3, 1});
        System.out.println(network.output(testInput2));

        INDArray testInput3 = Nd4j.create(new double[]{50, 60, 70}).reshape(new int[]{1, 3, 1});
        System.out.println(network.output(testInput3));
    }
}
