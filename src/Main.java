import model.*;
import preprocessing.dataImporter;


public class Main {
    public static void main(String[] args) {

        // RandomForest
        System.out.println("=============RandomForest Classification=============");
        RandomForest();

        System.out.println("=============RandomForestTuning Classification=============");
        RandomForestTuning();

        // OneR
        System.out.println("=============OneR Classification=============");
        OneR();

        // IBk
        System.out.println("=============IBK Classification=============");
        IBk();

        // Naive Bayes
        System.out.println("=============Naive Bayes Classification=============");
        NB();

        //J48
        System.out.println("=============J48 Classification=============");
        J48();
        System.out.println("=============J48 Tuning=============");
        J48Tuning();

        // SVM
        System.out.println("=============SVM Classification=============");
        SVM();

        // Logistic Regression
        System.out.println("=============Logistic Regression Classification=============");
        LR();

    }


    public static void RandomForest() {
        (new RandomForestClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void RandomForestTuning() {
        (new RandomForestTuning()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void OneR() {
        (new OneRClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void IBk() {
        (new IBkClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void NB() {
        (new NaiveBayesClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void J48() {
        (new J48Classifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void J48Tuning() {
        (new J48Tuning()).exec();
    }

    public static void SVM() {
        (new SVMClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void LR() {
        (new LogisticRegressionClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

}
