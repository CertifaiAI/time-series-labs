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
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class SalesDemandForecastIterator implements MultiDataSetIterator {

    private int batchSize;
    private int currentBatch=1;
    private int inputSequence;
    private int outputSequence;
    private int totalBatch ;
    private INDArray processedData;

    private List<Integer> data = new ArrayList<>();

    public SalesDemandForecastIterator(File dataset, int batchSize, int inputSequence, int outputSequence) throws IOException, InterruptedException {
        this.batchSize = batchSize;
        this.inputSequence = inputSequence;
        this.outputSequence = outputSequence;

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
        this.processedData = RecordConverter.toMatrix(transformedData);

//        this.totalBatch = this.processedData.size(0) - inputSequence + outputSequence + 1;
    }
    
    public void createBatchData (){

    }

    @Override
    public MultiDataSet next(int currentBatch) {
        //get encoder input
//        int sampleIndex = currentBatch - 1;
//        INDArray sample = this.processedData.get(sampleIndex);
//        this.currentBatch = this.currentBatch + 1;

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
        this.currentBatch = 1;
    }

    @Override
    public boolean hasNext() {
        if (this.currentBatch > this.totalBatch){
            return false;
        }else{
            return true;
        }
    }

    @Override
    public MultiDataSet next() {
        if (this.currentBatch <= this.totalBatch) {
            return next(this.currentBatch);
        }
        return null;
    }
}
