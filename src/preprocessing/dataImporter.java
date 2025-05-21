package preprocessing;

import weka.core.converters.ConverterUtils.DataSource;

public class dataImporter {
    public static DataSource trainSource;
    public static DataSource testSource;
    public static DataSource validSource;

    static {
        try {
            trainSource = new DataSource("data/KDDTrain.arff");
            testSource = new DataSource("data/KDDTest+.arff");
            validSource = new DataSource("data/KDDValid.arff");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
