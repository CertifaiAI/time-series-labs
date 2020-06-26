package regression.multivariate.customdataloader;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class MyCustomTimeSeriesIterator implements DataSetIterator {
    private List<INDArray> sequenceData;
    private int labelIdx;
    private int pointer;
    private long totalNumData;
    private long numBatches;
    private int timeStep;
    private int batchSize;
    private int numFeatures;
    private int numLabels;

    public MyCustomTimeSeriesIterator(INDArray data, int batchSize, int timeStep, int numFeatures, int labelIdx) {
        this.timeStep = timeStep;
        this.numFeatures = numFeatures;
        this.numLabels = 1;
        this.labelIdx = labelIdx;
        this.batchSize = batchSize;
        pointer = 0;
        totalNumData = data.size(0);
        sequenceData = generateListSequence(data);
        numBatches = sequenceData.size();
    }

    /**
     * Generate a list contains Array of sequence.
     * Each sequence contains the time step as stated by the user.
     * @param data
     * @return List<INDArray>
     */
    private List<INDArray> generateListSequence(INDArray data) {
        List<INDArray> dataSequence = new ArrayList<INDArray>();
        int maxIdxBeforeNull = (int) (totalNumData - timeStep);
        for (int i = 0; i <= maxIdxBeforeNull; i++) {
            dataSequence.add(data.get(NDArrayIndex.interval(i, i + timeStep)));
        }
        return dataSequence;
    }

    /**
     * Create and return a batch of data.
     * @param numOfSample
     * @return DataSet
     */
    @Override
    public DataSet next(int numOfSample) {
        //initialize arrays of features, labels, we will use this later to store a batch of data in the next step
        INDArray features = Nd4j.zeros(numOfSample, this.numFeatures, this.timeStep);
        INDArray labels = Nd4j.zeros(numOfSample, this.numLabels, this.timeStep);

        //initialize arrays of featureMask and labelMask, we will use this later when creating the dataset.
        //do note that featureMask consists of arrays of ones and labelMask consists of an array of zeros,
        //we will change the labelMask later.
        INDArray featureMask = Nd4j.ones(numOfSample, this.timeStep);
        INDArray labelMask = Nd4j.zeros(numOfSample, this.timeStep);

        //loop through the List<INDArray> created by the generateListSequence method.
        for (int i = 0; i < numOfSample; i++) {
            //get a sample of data (one row)
            INDArray sample = this.sequenceData.get(this.pointer);
            //slice out the feature array
            INDArray feature = sample.get(NDArrayIndex.all(), NDArrayIndex.interval(0, this.labelIdx)).transpose();
            //slice out the label array
            INDArray label = sample.get(NDArrayIndex.all(), NDArrayIndex.point(this.labelIdx));
            //store feature array into features that we had initialized earlier
            features.put(
                    new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()},
                    feature
            );
            //store labels array into labels that we had initialized earlier
            labels.put(
                    new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.all()},
                    label
            );
            this.pointer++;
        }

        /*
            we put 0s at all locations, except 1 at the index of label that we want to perform prediction.
            This is similar to the implementation of ALIGN_END in DL4J
            Example:
                location of label
                      |
            [0., 0., 1.]
         */
        labelMask.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(this.labelIdx)}, 1.0);

        DataSet dataSet = new DataSet(features, labels, featureMask, labelMask);
        return dataSet;
    }

    @Override
    public int inputColumns() {
        return this.numFeatures;
    }

    @Override
    public int totalOutcomes() {
        return 0;
    }


    @Override
    public List<String> getLabels() {
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
        this.pointer = 0;
    }

    @Override
    public int batch() {
        return this.batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {

    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return this.pointer < numBatches;
    }

    @Override
    public DataSet next() {
        if (this.hasNext()) {
            return next(this.batchSize);
        }
        return null;
    }

    @Override
    public void remove() {

    }

    @Override
    public void forEachRemaining(Consumer<? super DataSet> action) {

    }
}
