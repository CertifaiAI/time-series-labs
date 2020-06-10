package exercise.stockprediction;

import org.datavec.api.records.reader.RecordReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.List;

public class StockIterator implements MultiDataSetIterator {

    private int timeStep;
    private int batchSize;
    private int currentBatchIdx;
    private int labelIdx;
    private int stride;
    private RecordReader recordReaderData;
    private List<INDArray> sequenceData;

    public StockIterator(RecordReader recordReaderData, int timeStep, int batchSize, int labelIndex, int stride) {
        this.currentBatchIdx = 0 ;
        this.batchSize = batchSize; //timestep
        this.labelIdx = labelIndex;
        this.stride = stride;
        sequenceData = getSequenceData(recordReaderData);
    }

    private List<INDArray> getSequenceData(RecordReader data)
    {
        return null;
    }

    @Override
    public MultiDataSet next(int i) {
        return null;
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
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {

    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public MultiDataSet next() {
        return null;
    }
}
