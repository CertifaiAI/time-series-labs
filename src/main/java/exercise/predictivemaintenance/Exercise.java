package exercise.predictivemaintenance;

import org.datavec.api.transform.schema.Schema;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.io.IOException;

// Train a model to predict critical levels of asset given a time windows (multi-class classification)
// The critical level of 0 means asset is running fine,
// 2 means asset is going to fail soon, 15 cycles away from failure,
// 1 is intermediate level, 16-30 cycles away from failure.
// Given a time window, you are required to predict the time window belongs to class 0, 1, or 2.

// Use the dataset provided in "resources/datasets/predictivemaintenance"
// The train and test dataset consist of features (setting1, setting2, setting, s1, s2, ..., s21)
// and labels (Remaining Useful Life (RUL), label1, and label2)

// You should use only label2 and remove RUL and label1,
// since those labels are use for regression and binary classification task

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
