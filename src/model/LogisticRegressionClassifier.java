package model;

import preprocessing.dataImporter;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;  // Changed to CfsSubsetEval
import weka.attributeSelection.GreedyStepwise; // Changed to GreedyStepwise search

public class LogisticRegressionClassifier implements Command {
    public static void main(String[] args) {
        Command cmd = new LogisticRegressionClassifier();
        cmd.exec(dataImporter.trainSource, dataImporter.testSource);
    }

    @Override
    public void exec(DataSource trainSource, DataSource testSource) {
        try {
            System.out.println("Loading data...");
            // Load dataset
            Instances trainDataset = trainSource.getDataSet();
            Instances testDataset = testSource.getDataSet();

            // Set class index to the last attribute
            if (trainDataset.classIndex() == -1) {
                trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
            }

            if (testDataset.classIndex() == -1) {
                testDataset.setClassIndex(testDataset.numAttributes() - 1);
            }

            System.out.println("Converting string attributes to nominal...");
            // Convert string attributes to nominal if needed
            StringToNominal stringToNominal = new StringToNominal();
            stringToNominal.setAttributeRange("first-last"); // Convert all string attributes to nominal
            stringToNominal.setInputFormat(trainDataset);
            trainDataset = Filter.useFilter(trainDataset, stringToNominal);
            testDataset = Filter.useFilter(testDataset, stringToNominal);

            System.out.println("Removing useless attributes...");
            // Remove attributes with zero variance (useless for classification)
            RemoveUseless removeUseless = new RemoveUseless();
            removeUseless.setInputFormat(trainDataset);
            trainDataset = Filter.useFilter(trainDataset, removeUseless);
            testDataset = Filter.useFilter(testDataset, removeUseless);

            System.out.println("Converting nominal attributes to binary...");
            // Convert nominal attributes to binary for Logistic Regression
            NominalToBinary nominalToBinary = new NominalToBinary();
            nominalToBinary.setInputFormat(trainDataset);
            trainDataset = Filter.useFilter(trainDataset, nominalToBinary);
            testDataset = Filter.useFilter(testDataset, nominalToBinary);

            System.out.println("Normalizing attributes...");
            // Normalize numerical attributes
            Normalize normalize = new Normalize();
            normalize.setInputFormat(trainDataset);
            trainDataset = Filter.useFilter(trainDataset, normalize);
            testDataset = Filter.useFilter(testDataset, normalize);

            System.out.println("Original number of attributes: " + trainDataset.numAttributes());

            // Feature selection using CfsSubsetEval to avoid discretization issues
            System.out.println("Applying feature selection with CfsSubsetEval...");
            AttributeSelection attSelect = new AttributeSelection();
            CfsSubsetEval eval = new CfsSubsetEval();
            GreedyStepwise search = new GreedyStepwise();
            search.setSearchBackwards(true);
            attSelect.setEvaluator(eval);
            attSelect.setSearch(search);

            try {
                attSelect.SelectAttributes(trainDataset);
                // Apply selected attributes
                trainDataset = attSelect.reduceDimensionality(trainDataset);
                testDataset = attSelect.reduceDimensionality(testDataset);
                System.out.println("Reduced to " + trainDataset.numAttributes() + " attributes");
            } catch (Exception e) {
                System.out.println("Feature selection failed, continuing with all attributes: " + e.getMessage());
                // Continue without feature selection if it fails
            }

            // Check for class imbalance and apply SMOTE if necessary
            int[] classCounts = new int[trainDataset.numClasses()];
            for (int i = 0; i < trainDataset.numInstances(); i++) {
                classCounts[(int) trainDataset.instance(i).classValue()]++;
            }

            System.out.println("Class distribution before SMOTE:");
            for (int i = 0; i < classCounts.length; i++) {
                System.out.println("Class " + i + ": " + classCounts[i] + " instances");
            }

            // Only apply SMOTE if we have binary classification and imbalance
            if (trainDataset.numClasses() == 2) {
                double minorityRatio = Math.min(classCounts[0], classCounts[1]) /
                        (double) Math.max(classCounts[0], classCounts[1]);

                // Apply SMOTE if significant imbalance exists
                if (minorityRatio < 0.5) {
                    try {
                        System.out.println("Applying SMOTE for class imbalance (minority ratio: " + minorityRatio + ")...");
                        SMOTE smote = new SMOTE();
                        smote.setInputFormat(trainDataset);
                        trainDataset = Filter.useFilter(trainDataset, smote);

                        // Report new class distribution
                        classCounts = new int[trainDataset.numClasses()];
                        for (int i = 0; i < trainDataset.numInstances(); i++) {
                            classCounts[(int) trainDataset.instance(i).classValue()]++;
                        }
                        System.out.println("Class distribution after SMOTE:");
                        for (int i = 0; i < classCounts.length; i++) {
                            System.out.println("Class " + i + ": " + classCounts[i] + " instances");
                        }
                    } catch (Exception e) {
                        System.out.println("SMOTE failed, continuing without balancing: " + e.getMessage());
                    }
                }
            } else {
                System.out.println("Skipping SMOTE as this is not a binary classification problem");
            }

            // Create and configure Logistic Regression classifier
            Logistic lr = new Logistic();
            // Configure parameters
            lr.setRidge(0.5); // Regularization parameter
            lr.setMaxIts(100); // Maximum iterations

            // Build classifier
            System.out.println("Building Logistic Regression classifier...");
            lr.buildClassifier(trainDataset);
            System.out.println("LR parameters: " + String.join(" ", lr.getOptions()));

            // Evaluate model
            System.out.println("Evaluating Logistic Regression classifier...");
            Evaluation evaluation = new Evaluation(trainDataset);
            evaluation.evaluateModel(lr, testDataset);

            // Output evaluation results
            System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));
            System.out.println("Confusion Matrix:\n" + evaluation.toMatrixString());
            System.out.println("Correct % = " + evaluation.pctCorrect());
            System.out.println("Incorrect % = " + evaluation.pctIncorrect());
            System.out.println("AUC = " + evaluation.areaUnderROC(1));
            System.out.println("Kappa = " + evaluation.kappa());
            System.out.println("Precision = " + evaluation.precision(1));
            System.out.println("Recall = " + evaluation.recall(1));
            System.out.println("F-Measure = " + evaluation.fMeasure(1));
            System.out.println("Error Rate = " + evaluation.errorRate());
            System.out.println(evaluation.toClassDetailsString());

        } catch (Exception e) {
            System.out.println("Error in Logistic Regression classification: " + e.getMessage());
            e.printStackTrace();
        }
    }
}