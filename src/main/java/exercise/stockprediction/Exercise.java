package exercise.stockprediction;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.condition.column.ColumnCondition;
import org.datavec.api.transform.condition.column.NaNColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Exercise {

    public static void main(String[] args) throws IOException, InterruptedException {
        File data = new ClassPathResource("/datasets/StockData/AAPL.csv").getFile();

        Schema schema = new Schema.Builder()
                .addColumnString("Date") //Date,Open,High,Low,Close,Adj Close,Volume
                .addColumnDouble("Open")
                .addColumnDouble("High")
                .addColumnDouble("Low")
                .addColumnDouble("Close")
                .addColumnDouble("Adj Close")
                .addColumnLong("Volume")
                .build();

        TransformProcess tp = new TransformProcess.Builder(schema)
                .stringToTimeTransform("Date", "yyyy-MM-dd", DateTimeZone.UTC)
                .transform(new DeriveColumnsFromTimeTransform.Builder("Date")
                        .addIntegerDerivedColumn("year", DateTimeFieldType.year())
                        .addIntegerDerivedColumn("month", DateTimeFieldType.monthOfYear())
                        .addIntegerDerivedColumn("day", DateTimeFieldType.dayOfMonth())
                        .addIntegerDerivedColumn("dayOfWeek", DateTimeFieldType.dayOfWeek())
                        .build())
                .removeColumns("Date")
                .build();

        int numLinesToSkip = 1;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(data));

        List<List<Writable>> originalData = new ArrayList<>();
        while (recordReader.hasNext()) {
            originalData.add(recordReader.next());
        }

        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, tp);
        //Create iterator from processedData
        RecordReader collectionRecordReader = new CollectionRecordReader(processedData);
//        DataSetIterator iterator = new RecordReaderDataSetIterator(collectionRecordReader,2,4,2);
        while(collectionRecordReader.hasNext())
        {
            System.out.println(collectionRecordReader.next());
        }

    }
}
