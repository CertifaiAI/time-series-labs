package regression.seq2seq;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.sequence.ConvertToSequence;
import org.datavec.api.transform.sequence.comparator.NumericalColumnComparator;
import org.datavec.api.transform.sequence.comparator.StringComparator;
import org.datavec.api.transform.transform.sequence.SequenceOffsetTransform;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
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
import org.joda.time.DateTimeComparator;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public class Seq2SeqSalesDemandForecast {
    static int batchSize;
    public static void main(String[] args) throws IOException, InterruptedException {
        File dataset = new ClassPathResource("/datasets/sales_demand_forecast.csv").getFile();

        Schema inputDataSchema = new Schema.Builder()
                .addColumnString("date")
                .addColumnInteger("sales")
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .stringToTimeTransform("date","YYYY-MM-DD", DateTimeZone.UTC)
                .transform(new DeriveColumnsFromTimeTransform.Builder("date")
                        .addIntegerDerivedColumn("month", DateTimeFieldType.monthOfYear())
                        .addIntegerDerivedColumn("day", DateTimeFieldType.dayOfMonth())
                        .addIntegerDerivedColumn("dayOfWeek", DateTimeFieldType.dayOfWeek())
                        .build())
                .removeColumns("date")
                .build();

        int numLinesToSkip = 1;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(dataset));

        List<List<Writable>> originalData = new ArrayList<>();
        while(recordReader.hasNext()){
            originalData.add(recordReader.next());
        }

        List<List<Writable>> transformedData = LocalTransformExecutor.execute(originalData, tp);
        INDArray processedDataArray = RecordConverter.toMatrix(transformedData);

        List<INDArray> trainTestList =  trainTestSplit(processedDataArray, 0.9);

        SalesDemandIterator trainIter = new SalesDemandIterator(trainTestList.get(0), 32, 10, 5);
        SalesDemandIterator testIter = new SalesDemandIterator(trainTestList.get(1), 16, 10, 5);

        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("encoderInput", "decoderInput")
                .addLayer("encoder", new LSTM.Builder().nIn(3).nOut(32).activation(Activation.TANH).build(), "encoderInput")
                .addVertex("lastTimeStep", new LastTimeStepVertex("encoderInput"), "encoder")
                .addVertex("encoderState", new DuplicateToTimeSeriesVertex("decoderInput"), "lastTimeStep")
                .addVertex("merge", new MergeVertex(), "decoderInput", "encoderState")
                .addLayer("decoder", new LSTM.Builder().nIn(32 + 1).nOut(32).activation(Activation.TANH).build(), "merge")
                .addLayer("output", new RnnOutputLayer.Builder().nIn(32).nOut(1).activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build(), "decoder")
                .setOutputs("output").build();

        ComputationGraph graph = new ComputationGraph(conf2);
        graph.init();

        // training
        graph.setListeners(new ScoreIterationListener(10));

        graph.fit(trainIter, 10);

        while (testIter.hasNext()){
            MultiDataSet testBatch = testIter.next();

            INDArray encoderInput = testBatch.getFeatures()[0];
            INDArray decoderInput = testBatch.getFeatures()[1];
            INDArray label = testBatch.getLabels()[0];

            INDArray decoderInputStart = Nd4j.zeros(16, 1, 1);
            decoderInputStart.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(0)},
                    decoderInput.get(NDArrayIndex.all(), NDArrayIndex.point(0), NDArrayIndex.point(0)));

            INDArray prediction = predict(graph, encoderInput, decoderInputStart);
            INDArray mse = mseError(prediction, label);
            System.out.println(mse.getDouble());
        }
    }

    private static INDArray mseError(INDArray prediction, INDArray label){
        INDArray a = Nd4j.toFlattened(prediction);
        INDArray b = Nd4j.toFlattened(label);

        return Transforms.pow(a.sub(b),2).mean();
    }

    private static INDArray predict(ComputationGraph graph, INDArray encoderInput, INDArray decoderInput){
        INDArray encState = graph.feedForward(new INDArray[]{encoderInput, decoderInput}, false, false).get("encoderState");
        org.deeplearning4j.nn.layers.recurrent.LSTM decoder = (org.deeplearning4j.nn.layers.recurrent.LSTM) graph.getLayer("decoder");
        Layer output = graph.getLayer("output");
        GraphVertex mergeVertex = graph.getVertex("merge");
        INDArray prediction = null;

        for (int i = 0; i < 5; i++) {
            mergeVertex.setInputs(decoderInput, encState);
            INDArray merged = mergeVertex.doForward(false, LayerWorkspaceMgr.noWorkspaces());
            INDArray decOutput = decoder.rnnTimeStep(merged, LayerWorkspaceMgr.noWorkspaces());
            INDArray out = output.activate(decOutput, false, LayerWorkspaceMgr.noWorkspaces());

            decoderInput = out;

            if(prediction == null){
                prediction = out;
            }else {
                prediction = Nd4j.concat(2, prediction, out);
            }
        }

        return prediction;
    }

    private static List<INDArray> trainTestSplit(INDArray input, double trainPerc){

        long sampleSize = input.shape()[0];
        int trainSize = (int)Math.floor(sampleSize * trainPerc);

        INDArray trainArray = input.get(NDArrayIndex.interval(0,trainSize), NDArrayIndex.all());
        INDArray testArray = input.get(NDArrayIndex.interval(trainSize, sampleSize), NDArrayIndex.all());

        return Arrays.asList(trainArray, testArray);
    }
}
