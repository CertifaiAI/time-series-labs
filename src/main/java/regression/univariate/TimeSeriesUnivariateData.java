package regression.univariate;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

public class TimeSeriesUnivariateData {

    private double[] timeSeriesSequence;
    private int timeStep;
    private ArrayList<INDArray> data;

    public TimeSeriesUnivariateData(double[] data, int timeStep)
    {
        this.timeSeriesSequence = data;
        this.timeStep = timeStep;
        this.data = splitSequence(data, timeStep);
    }

    private ArrayList<INDArray> splitSequence(double[] inputArray, int nSteps) {
        ArrayList<INDArray> result = new ArrayList<>();
        List<List<Double>> X = new ArrayList<>();
        List<Double> y = new ArrayList<>();

        for (int i = 0; i < inputArray.length; i++) {
            int endIndex = i + nSteps;
            if (endIndex >= inputArray.length - 1) {
                break;
            }
            ArrayList<Double> sequenceX = new ArrayList<Double>();
            for (int j = i; j < endIndex; j++) {
                sequenceX.add(inputArray[j]);
            }
            double sequenceY = inputArray[endIndex];
            X.add(sequenceX);
            y.add(sequenceY);
        }

        double[][] outputX = convertNestedListToArray(X);
        double[] outputY = y.stream().mapToDouble(Double::doubleValue).toArray();

        INDArray outputXNDArray = Nd4j.create(outputX);
        INDArray outputYNDArray = Nd4j.create(outputY);

        result.add(outputXNDArray);
        result.add(outputYNDArray);
        return result;
    }

    private double[][] convertNestedListToArray(List<List<Double>> input) {
        double[][] array = new double[input.size()][];
        int timeStep = input.get(0).size();

        for (int i = 0; i < input.size(); i++) {
            double[] blankArray = new double[timeStep];
            List<Double> currList = input.get(i);
            for (int j = 0; j < timeStep; j++) {
                blankArray[j] = currList.get(j);
            }
            array[i] = blankArray;
        }
        return array;
    }

    public INDArray getFeatureMatrix() {
        INDArray feature = data.get(0);
        //reshape from [samples, timesteps] into [samples, timesteps, num of features], here the number of feature is 1
        INDArray expandedDimArray = Nd4j.expandDims(feature, 2);
        return expandedDimArray;
    }

    public INDArray getLabels() {
        INDArray label = data.get(1);
        //reshape from [samples, ] into [samples, timesteps, features]
        INDArray expdim1 = Nd4j.expandDims(label, 1);
        INDArray expdim2 = Nd4j.expandDims(expdim1, 2);
        return expdim2;
    }

    public int getSequenceLength(){
        return (int) getFeatureMatrix().shape()[1];
    }

}
