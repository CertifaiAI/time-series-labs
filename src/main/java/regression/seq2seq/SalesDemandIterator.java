package regression.seq2seq;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SalesDemandIterator implements MultiDataSetIterator {

    private final int batchSize;
    private final int inputSequence;
    private final int outputSequence;
    private final int totalBatch;
    private final long featureSize;
    private final int labelIndex;
    List<INDArray> sequenceData;
    private int currentBatch = 0;


    public SalesDemandIterator(INDArray data, int batchSize, int inputSequence, int outputSequence) {
        this.batchSize = batchSize;
        this.inputSequence = inputSequence;
        this.outputSequence = outputSequence;
        this.featureSize = 4;
        this.labelIndex = 4;

        this.sequenceData = createSequence(data);
        this.totalBatch = this.sequenceData.size() / batchSize;
    }

    // organize data into sequences
    private List<INDArray> createSequence(INDArray data) {
        List<INDArray> sequenceData = new ArrayList<>();
        int sequenceLength = this.inputSequence + this.outputSequence;
        long dataLength = data.size(0);

        long totalSample = dataLength - sequenceLength + 1;

        for (int i = 0; i < totalSample; i++) {
            sequenceData.add(data.get(NDArrayIndex.interval(i, i + sequenceLength), NDArrayIndex.all()));
        }
        return sequenceData;
    }

    // prepare input data for encoder and decoder and decoder label
    // decoder input = y, y+1, y+2, ..., y+n
    // decoder label = y+1, y+2, y+3, ..., y+n
    private List<INDArray> prepareSample(INDArray sample) {
        INDArray label = sample.get(NDArrayIndex.all(), NDArrayIndex.point(this.labelIndex));
        // exclude year column
        INDArray feature = sample.get(NDArrayIndex.all(), NDArrayIndex.interval(1, this.labelIndex+1));

        INDArray encoderInput = feature.get(NDArrayIndex.interval(0, this.inputSequence), NDArrayIndex.all()).permute(1, 0);
        INDArray decoderInput = label.get(
                NDArrayIndex.interval(this.inputSequence - 1, this.inputSequence + this.outputSequence - 1)
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

        for (int i = 0; i < this.batchSize; i++) {
            int sampleIndex = currentBatch * this.batchSize + i;
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
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {

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
        return this.currentBatch < this.totalBatch;
    }

    @Override
    public MultiDataSet next() {
        if (hasNext()) {
            return next(this.currentBatch);
        }
        return null;
    }

    public MultiDataSet getLastSequence(){
        INDArray encoderInput = Nd4j.zeros(1, this.featureSize, this.inputSequence);
        INDArray decoderInput = Nd4j.zeros(1, 1, this.outputSequence);
        INDArray label = Nd4j.zeros(1, 1, this.outputSequence);

        INDArray inputMask = Nd4j.ones(1, this.inputSequence);
        INDArray labelMask = Nd4j.ones(1, this.outputSequence);

        INDArray lastSample = this.sequenceData.get(this.sequenceData.size()-1);
        List<INDArray> lastSamplePrepared = prepareSample(lastSample);

        encoderInput.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()}, lastSamplePrepared.get(0));
        decoderInput.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()}, lastSamplePrepared.get(1));
        label.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all()}, lastSamplePrepared.get(2));

        return new MultiDataSet(new INDArray[]{encoderInput, decoderInput}, new INDArray[]{label},
                new INDArray[]{inputMask, labelMask}, new INDArray[]{labelMask});
    }
}
