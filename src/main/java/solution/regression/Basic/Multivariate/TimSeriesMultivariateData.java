package solution.regression.Basic.Multivariate;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;

public class TimSeriesMultivariateData {
    private List<INDArray> inputSequenceList;
    private List<INDArray> outputSequenceList;
    private INDArray data;
    private List<INDArray> splitSequenceOutput; // a list that contains both feature and label
    private INDArray feature;
    private INDArray label;


    public TimSeriesMultivariateData(List<INDArray> input, List<INDArray> output, int timeStep) {
        this.inputSequenceList = new ArrayList<>();
        this.outputSequenceList = new ArrayList<>();
        this.splitSequenceOutput = new ArrayList<>();
        for (INDArray inputArray : input) {
            //convert (row,) to (row, column) data structure
            INDArray reshapedInput = inputArray.reshape(new int[] {(int) inputArray.length(), 1});
            this.inputSequenceList.add(reshapedInput);
        }

        for (INDArray outputArray : output) {
            INDArray reshapedOutput = outputArray.reshape(new int[]{(int) outputArray.shape()[0], 1});
            this.outputSequenceList.add(reshapedOutput);
        }

        INDArray stackedInput =  Nd4j.hstack(this.inputSequenceList);
        INDArray stackedOutput =  Nd4j.hstack(this.outputSequenceList);
        this.data = Nd4j.hstack(stackedInput, stackedOutput);
        this.splitSequenceOutput = splitSequence(this.data, timeStep);
        this.feature = this.splitSequenceOutput.get(0);
        this.label = this.splitSequenceOutput.get(1);
    }


    private List<INDArray> splitSequence(INDArray input, int steps)
    {
        double[] X = new double[(int) input.length()];
        double[] Y = new double[(int) input.length()];
        List<INDArray> resultsX = new ArrayList<>();
        List<INDArray> resultsY = new ArrayList<>();
        List<INDArray> finalResults = new ArrayList<>();


        int numRows = (int) input.shape()[0];
        int numCols = (int) input.shape()[1];

        for (int i = 0; i < numRows; i++)
        {
            int endIndex = i + steps;
            if (endIndex > numRows) {break;}
            INDArray seqX = input.get(NDArrayIndex.interval(i, endIndex), NDArrayIndex.interval(0, numCols-1));
            INDArray seqY = input.get(NDArrayIndex.point(endIndex-1),NDArrayIndex.point(numCols-1));
            resultsX.add(seqX);
            resultsY.add(seqY);

        }

        // number of samples
        assert(resultsX.size() == resultsY.size());
        int sampleSize = resultsX.size();
        int numFeatures = (int) resultsX.get(0).shape()[1];
        int outputCols = 1;
        INDArray finalResultsX = Nd4j.create(resultsX, new int[]{sampleSize, steps, numFeatures});
        finalResultsX.setShapeAndStride(new int[]{sampleSize, steps, numFeatures}, new int[] {1, sampleSize, sampleSize*steps});
        INDArray finalResultsY = Nd4j.create(resultsY, new int[]{sampleSize, outputCols,1});

        finalResults.add(finalResultsX);
        finalResults.add(finalResultsY);

        return finalResults;
    }

    public String toString()
    {
        return String.valueOf(data);
    }

    public INDArray getFeature() {
        return feature;
    }

    public INDArray getLabel() {
        return label;
    }



}
