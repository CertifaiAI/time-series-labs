package solution.regression.Basic.Multivariate;


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

import java.util.ArrayList;
import java.util.List;

/**
 * Multivariate time series, Multi-in single out
 *
 *
 * [[   10.0000,   15.0000,   25.0000],
 *  [   20.0000,   25.0000,   45.0000],
 *  [   30.0000,   35.0000,   65.0000],
 *  [   40.0000,   45.0000,   85.0000],
 *  [   50.0000,   55.0000,  105.0000],
 *  [   60.0000,   65.0000,  125.0000],
 *  [   70.0000,   75.0000,  145.0000],
 *  [   80.0000,   85.0000,  165.0000],
 *  [   90.0000,   95.0000,  185.0000]]
 *
 *  Example:
 *  Input:
 *  [[10.000, 15.000],
 *   [20.000, 25.000]]
 *
 *  Output:
 *  [[65.000]]
 */

public class MultivariateLSTM {

    private static double learningRate = 0.001;

    public static void main(String[] args) {

        INDArray seq1 = Nd4j.create(new double[]{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0});
        INDArray seq2 = Nd4j.create(new double[]{15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0});
        INDArray outputSeq = seq1.addRowVector(seq2);

        List<INDArray> allInput = new ArrayList<>();
        allInput.add(seq1);
        allInput.add(seq2);

        List<INDArray> allOutput = new ArrayList<>();
        allOutput.add(outputSeq);

        TimSeriesMultivariateData data = new TimSeriesMultivariateData(allInput, allOutput, 3);
        System.out.println(data.getFeature());
        System.out.println(data.getLabel());

        INDArray feature = data.getFeature().permute(0,2,1);
        INDArray label = data.getLabel();

        System.out.println("feature size: " + feature.shapeInfoToString());
        System.out.println("label size: " + label.shapeInfoToString());

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(learningRate))
                .weightInit(WeightInit.XAVIER)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .miniBatch(false)
                .list()
                .layer(0, new LSTM.Builder()
                        .nIn(2)
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
        MultiLayerNetwork network = new MultiLayerNetwork(config);
        network.init();

        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);
        network.setListeners(new StatsListener(storage, 10));
        for (int i = 0; i < 10000; i++) {
            network.fit(feature, label);
        }


    }
}
