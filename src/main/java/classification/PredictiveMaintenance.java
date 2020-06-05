package classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionSequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.sequence.comparator.NumericalColumnComparator;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PredictiveMaintenance {
    static int sequenceLength = 15;

    public static void main(String[] args) throws IOException, InterruptedException {
        File trainDataset = new ClassPathResource("/datasets/predictivemaintenance/train.csv").getFile();

        Schema inputDataSchema = new Schema.Builder()
                .addColumnsInteger("id","cycle")
                .addColumnsDouble("setting1","setting2","setting3",
                        "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
                        "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21","cycle_norm"
                        )
                .addColumnsInteger("RUL","label1","label2")
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .removeColumns("setting3","s1","s5","s10","s16","s18","s19","RUL","label2")
                .convertToSequence(Arrays.asList("id"), new NumericalColumnComparator("cycle"))
                .build();

        int numLinesToSkip = 1;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(trainDataset));

        List<List<Writable>> originalData = new ArrayList<>();
        while (recordReader.hasNext()) {
            originalData.add(recordReader.next());
        }

        List<List<List<Writable>>> processedData = LocalTransformExecutor.executeToSequence(originalData, tp);

        List<List<List<Writable>>> data = new ArrayList<>();
        for (List<List<Writable>> engineSample : processedData) {
            for (int j = 0; j < (engineSample.size() - sequenceLength); j++) {
                data.add(engineSample.subList(j, j + sequenceLength));
            }
        }

        SequenceRecordReader sequenceRecordReader = new CollectionSequenceRecordReader(data);

        SequenceRecordReaderDataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(sequenceRecordReader, 32, 2, 20);

        DataNormalization scaler = new NormalizerMinMaxScaler();
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);

        int numInput = trainIter.inputColumns();
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .l2(0.00001)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .graphBuilder()
                .addInputs("input")
                .addLayer("lstm", new LSTM.Builder()
                                .nIn(numInput)
                                .nOut(100)
                                .activation(Activation.TANH)
                                .build(),
                        "input")
                .addLayer("lstm2", new LSTM.Builder()
                                .nIn(100)
                                .nOut(50)
                                .activation(Activation.TANH)
                                .build(),
                        "lstm")
                .addLayer("output", new RnnOutputLayer.Builder()
                                .nIn(50)
                                .nOut(2)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .build(),
                        "lstm2")
                .setOutputs("output")
                .build();

        ComputationGraph model = new ComputationGraph(config);
        model.init();
        model.setListeners(new ScoreIterationListener( 10));

        /*
		#### LAB STEP 4 #####
		Train the model
        */
        for (int i=0; i<10; i++){
            System.out.println("EPOCH: " + i);
            model.fit(trainIter);

            Evaluation eval = model.evaluate(trainIter);
            System.out.println(eval.stats());
        }

//        System.out.println("=== BEFORE ===");

//        for (int i=0;i<originalData.size();i++) {
//            System.out.println(originalData.get(i));
//        }
//
//        System.out.println("=== AFTER ===");
//        for (int i=0;i<processedData.size();i++) {
//            System.out.println(processedData.get(i));
//        }
    }
}
