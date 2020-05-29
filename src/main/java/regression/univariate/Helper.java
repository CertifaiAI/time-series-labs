package regression.univariate;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Helper {
    public static INDArray expandDimsHelper(INDArray input)
    {
        // expand at the 0th index
        INDArray expand1 = Nd4j.expandDims(input, 0);
        INDArray expand2 = Nd4j.expandDims(expand1, 2);
        return expand2;
    }
}
