package model;

import preprocessing.dataImporter;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.rules.OneR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;

public class OneRClassifier implements Command {
    public static void main(String[] args) {
        // Fix: Create an instance of OneRClassifier, not LogisticRegressionClassifier
        Command cmd = new OneRClassifier();
        cmd.exec(dataImporter.trainSource, dataImporter.testSource);
    }

    @Override
    public void exec(DataSource trainSource, DataSource testSource) {
        try {
            // Load dataset
            Instances trainDataset = trainSource.getDataSet();

            // Load testing dataset
            Instances testDataset = testSource.getDataSet();

            // Set class index to the last attribute (assuming the last attribute is the class label)
            if (trainDataset.classIndex() == -1) {
                trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
            }

            if (testDataset.classIndex() == -1) {
                testDataset.setClassIndex(testDataset.numAttributes() - 1);
            }

            // Convert any string attributes to nominal if needed (OneR requires nominal attributes)
            StringToNominal stringToNominal = new StringToNominal();
            stringToNominal.setAttributeRange("first-last"); // Convert all attributes
            stringToNominal.setInputFormat(trainDataset);
            trainDataset = Filter.useFilter(trainDataset, stringToNominal);
            testDataset = Filter.useFilter(testDataset, stringToNominal);

            // Create and train the OneR classifier
            OneR oner = new OneR();
            // You can set the minimum bucket size (default is 6)
            oner.setMinBucketSize(6);
            oner.buildClassifier(trainDataset);

            System.out.println("OneR classifier built successfully");
            System.out.println("OneR params: " + String.join(" ", oner.getOptions()));
            System.out.println("OneR model: \n" + oner);

            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(oner, testDataset);

            // Output the evaluation results
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));

            // Print the confusion matrix
            System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

            // Print additional evaluation metrics
            System.out.println("Correct % = " + eval.pctCorrect());
            System.out.println("Incorrect % = " + eval.pctIncorrect());
            System.out.println("AUC = " + eval.areaUnderROC(1));
            System.out.println("Kappa = " + eval.kappa());
            System.out.println("MAE = " + eval.meanAbsoluteError());
            System.out.println("RMSE = " + eval.rootMeanSquaredError());
            System.out.println("RAE = " + eval.relativeAbsoluteError());
            System.out.println("RRSE = " + eval.rootRelativeSquaredError());
            System.out.println("Precision = " + eval.precision(1));
            System.out.println("Recall = " + eval.recall(1));
            System.out.println("F-Measure = " + eval.fMeasure(1));
            System.out.println("Error Rate = " + eval.errorRate());
            System.out.println(eval.toClassDetailsString());

        } catch (Exception e) {
            System.out.println("Error in OneR classification:");
            e.printStackTrace();
        }
    }
}