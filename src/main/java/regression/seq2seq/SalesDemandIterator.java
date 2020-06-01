package regression.seq2seq;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public class SalesDemandIterator implements MultiDataSetIterator {

    private int batchSize;
    private int currentBatch=0;
    private int inputSequence;
    private int outputSequence;
    private int totalBatch ;
    private long featureSize;
    List<INDArray> sequenceData;

//    private List<Integer> data = new ArrayList<>();

    public SalesDemandIterator(INDArray data, int batchSize, int inputSequence, int outputSequence) throws IOException, InterruptedException {
        this.batchSize = batchSize;
        this.inputSequence = inputSequence;
        this.outputSequence = outputSequence;
        this.featureSize = data.size(1) - 1;

        this.sequenceData = createSequence(data);
        this.totalBatch = this.sequenceData.size() / batchSize;


    }
    
    private List<INDArray> createSequence (INDArray data){
        List<INDArray> sequenceData = new ArrayList<>();
        int sequenceLength = this.inputSequence + this.outputSequence;
        long dataLength = data.size(0);

        long totalSample = dataLength - sequenceLength + 1;

        for(int i = 0; i<totalSample;i++){
            sequenceData.add(data.get(NDArrayIndex.interval(i,i+sequenceLength),NDArrayIndex.all()));
        }
        return sequenceData;
    }

    private List<INDArray> prepareSample(INDArray sample){
        INDArray label = sample.get(NDArrayIndex.all(), NDArrayIndex.point(3));
        INDArray feature = sample.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 3));

        INDArray encoderInput = feature.get(NDArrayIndex.interval(0,this.inputSequence), NDArrayIndex.all()).permute(1,0);
        INDArray decoderInput = label.get(
                NDArrayIndex.interval(this.inputSequence-1, this.inputSequence + this.outputSequence - 1)
        );
        INDArray decoderLabel = label.get(
                NDArrayIndex.interval(this.inputSequence, this.inputSequence + this.outputSequence)
        );

        return Arrays.asList(encoderInput, decoderInput, decoderLabel);
    }

    @Override
    public MultiDataSet next(int currentBatch) {
        INDArray encoderInput = Nd4j.zeros(this.batchSize, this.featureSize, this.inputSequence);
        INDArray decoderInput = Nd4j.zeros(this.batchSize, 1, this.outputSequence);
        INDArray label = Nd4j.zeros(this.batchSize, 1, this.outputSequence);

        INDArray inputMask = Nd4j.ones(this.batchSize, this.inputSequence);
        INDArray labelMask = Nd4j.ones(this.batchSize, this.outputSequence);

        for(int i =0;i<this.batchSize;i++){
            int sampleIndex = currentBatch*this.batchSize+i;
            List<INDArray> sample = prepareSample(this.sequenceData.get(sampleIndex));

            encoderInput.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()}, sample.get(0));
            decoderInput.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()}, sample.get(1));
            label.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()}, sample.get(2));
        }

        this.currentBatch = this.currentBatch + 1;

        return new MultiDataSet(new INDArray[]{encoderInput, decoderInput}, new INDArray[]{label},
                new INDArray[]{inputMask, labelMask}, new INDArray[]{labelMask});
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {

    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        this.currentBatch = 0;
    }

    @Override
    public boolean hasNext() {
        if (this.currentBatch >= this.totalBatch){
            return false;
        }else{
            return true;
        }
    }

    @Override
    public MultiDataSet next() {
        if(hasNext()){
            return next(this.currentBatch);
        }
        return null;
    }
}
