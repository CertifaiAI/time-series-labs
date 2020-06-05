package exercise.predictivemaintenance;

import org.datavec.api.transform.schema.Schema;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.io.IOException;

public class Exercise {

    public static void main(String[] args) throws IOException {
        File trainDataset = new ClassPathResource("/datasets/predictivemaintenance/train.csv").getFile();
        File testDataset = new ClassPathResource("/datasets/predictivemaintenance/test.csv").getFile();

        Schema inputDataSchema = new Schema.Builder()
                .addColumnsInteger("id", "cycle")
                .addColumnsDouble("setting1", "setting2", "setting3",
                        "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
                        "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "cycle_norm"
                )
                .addColumnsInteger("RUL", "label1", "label2")
                .build();
    }
}
