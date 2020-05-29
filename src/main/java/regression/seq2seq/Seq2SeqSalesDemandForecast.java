package regression.seq2seq;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.sequence.ConvertToSequence;
import org.datavec.api.transform.sequence.comparator.NumericalColumnComparator;
import org.datavec.api.transform.sequence.comparator.StringComparator;
import org.datavec.api.transform.transform.sequence.SequenceOffsetTransform;
import org.datavec.api.transform.transform.time.DeriveColumnsFromTimeTransform;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.joda.time.DateTimeComparator;
import org.joda.time.DateTimeFieldType;
import org.joda.time.DateTimeZone;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public class Seq2SeqSalesDemandForecast {
    static int batchSize;
    public static void main(String[] args) throws IOException, InterruptedException {
        File dataset = new ClassPathResource("/datasets/sales_demand_forecast.csv").getFile();

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
//                .convertToSequence(Arrays.asList("year","month","day"), new NumericalColumnComparator("sales", true))
//                .convertToSequence(true, Arrays.asList("date"), new NumberComparator("date"))
                .removeColumns("date")
//                .convertToSequence(Arrays.asList("date"), new DateComparator("date"))
//                .offsetSequence(Arrays.asList("sales"),1, SequenceOffsetTransform.OperationType.NewColumn)
                .build();

        int numLinesToSkip = 1;
        char delimiter = ',';
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(dataset));

        List<List<Writable>> originalData = new ArrayList<>();
        while(recordReader.hasNext()){
            originalData.add(recordReader.next());
        }



        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, tp);

        //create sequential data
        INDArray test = RecordConverter.toMatrix(processedData);


        System.out.println("=== BEFORE ===");

//        for (int i=0;i<originalData.size();i++) {
//            System.out.println(originalData.get(i));
//        }

        System.out.println("=== AFTER ===");
        for (int i=0;i<processedData.size();i++) {
            System.out.println(processedData.get(i));
        }
    }
}
