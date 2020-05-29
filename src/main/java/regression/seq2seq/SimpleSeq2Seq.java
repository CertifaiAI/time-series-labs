package regression.seq2seq;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class SimpleSeq2Seq {
    static int nFeatures = 1;
    static int nHidden = 15;
    static int nOutput = 1;

    static int epochs = 400;

    static Logger log = LoggerFactory.getLogger(SimpleSeq2Seq.class);

    public static void main(String[] args) {
        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.1))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("encoderInput", "decoderInput")
                .addLayer("encoder", new LSTM.Builder().nIn(nFeatures).nOut(nHidden).activation(Activation.TANH).build(), "encoderInput")
                .addVertex("lastTimeStep", new LastTimeStepVertex("encoderInput"), "encoder")
                .addVertex("encoderState", new DuplicateToTimeSeriesVertex("decoderInput"), "lastTimeStep")
                .addVertex("merge", new MergeVertex(), "decoderInput", "encoderState")
                .addLayer("decoder", new LSTM.Builder().nIn(nHidden + nFeatures).nOut(nHidden).activation(Activation.TANH).build(), "merge")
                .addLayer("output", new RnnOutputLayer.Builder().nIn(nHidden).nOut(nOutput).activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build(), "decoder")
                .setOutputs("output").build();

        ComputationGraph graph = new ComputationGraph(conf2);
        graph.init();

        // training
        MultiDataSet dataSet = getDataset();
        train(graph, epochs, dataSet);

        // testing
        INDArray encoderInputTest = Nd4j.create(new double[][][]{{{2, 3, 4}}});
        INDArray decoderInputTest = Nd4j.create(new double[][][]{{{0}}});
        test(graph, encoderInputTest, decoderInputTest);
    }

    static void train(ComputationGraph graph, int epochs, MultiDataSet dataSet) {
        graph.setListeners(new ScoreIterationListener(10));

        for (int epoch = 0; epoch < epochs; epoch++) {
            graph.fit(dataSet);
        }
    }

    static void test(ComputationGraph graph, INDArray encoderInput, INDArray decoderInput) {
        INDArray encState = graph.feedForward(new INDArray[]{encoderInput, decoderInput}, false, false).get("encoderState");
        org.deeplearning4j.nn.layers.recurrent.LSTM decoder = (org.deeplearning4j.nn.layers.recurrent.LSTM) graph.getLayer("decoder");
        Layer output = graph.getLayer("output");
        GraphVertex mergeVertex = graph.getVertex("merge");

        for (int i = 0; i < 3; i++) {
            mergeVertex.setInputs(decoderInput, encState);
            INDArray merged = mergeVertex.doForward(false, LayerWorkspaceMgr.noWorkspaces());
            INDArray decOutput = decoder.rnnTimeStep(merged, LayerWorkspaceMgr.noWorkspaces());
            INDArray out = output.activate(decOutput, false, LayerWorkspaceMgr.noWorkspaces());

            decoderInput = out;

            log.info("y+" + (i + 1) + " forecast = " + out.getDouble());
        }
    }

    static MultiDataSet getDataset() {
        INDArray encoderInput = Nd4j.create(
                new double[][][]{
                        {{1, 2, 3}},
                        {{2, 3, 4}},
                        {{3, 4, 5}},
                        {{4, 5, 6}}}
        );

        INDArray decoderInput = Nd4j.create(
                new double[][][]{
                        {{0, 4, 5, 6}},
                        {{0, 5, 6, 7}},
                        {{0, 6, 7, 8}},
                        {{0, 7, 8, 9}}}
        );

        INDArray label = Nd4j.create(
                new double[][][]{
                        {{4, 5, 6, 0}},
                        {{5, 6, 7, 0}},
                        {{6, 7, 8, 0}},
                        {{7, 8, 9, 0}}}
        );

        INDArray inputMask = Nd4j.ones(4, 3);
        INDArray labelMask = Nd4j.ones(4, 4);

        return new MultiDataSet(new INDArray[]{encoderInput, decoderInput}, new INDArray[]{label},
                new INDArray[]{inputMask, labelMask}, new INDArray[]{labelMask});
    }
}
