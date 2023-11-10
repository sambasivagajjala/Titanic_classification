import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;

public class TitanicSurvivalPrediction {

    public static void main(String[] args) {
        try {
            // Load dataset
            DataSource source = new DataSource("path/to/titanic.arff");
            Instances data = source.getDataSet();

            // Set the class attribute
            data.setClassIndex(data.numAttributes() - 1);

            // Build a decision tree classifier
            J48 tree = new J48();
            tree.buildClassifier(data);

            // Make predictions for a new instance
            Instances testInstance = new Instances("TestInstance", data.attributeNames(), 1);
            testInstance.setClassIndex(testInstance.numAttributes() - 1);
            // Set values for socio-economic status, age, gender, etc.
            // testInstance.setValue(data.attribute("SocioEconomicStatus"), value);
            // testInstance.setValue(data.attribute("Age"), value);
            // testInstance.setValue(data.attribute("Gender"), value);
            // ...

            double prediction = tree.classifyInstance(testInstance.firstInstance());

            System.out.println("Predicted survival: " + data.classAttribute().value((int) prediction));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}