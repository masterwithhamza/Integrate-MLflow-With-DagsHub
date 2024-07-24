import argparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

def train_and_log_model(max_iter):
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up MLflow experiment
    mlflow.set_experiment("Iris Experiment")

    # Start MLflow run
    with mlflow.start_run() as run:
        # Train the Logistic Regression model
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log parameters
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_param("max_iter", max_iter)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Infer and log signature
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)

        # Register the model
        client = MlflowClient()
        model_name = "IrisLogisticRegressionModel"
        model_uri = f"runs:/{run.info.run_id}/model"

        try:
            # Check if the model is already registered
            client.get_registered_model(model_name)
        except Exception:
            # Register model if it doesn't exist
            client.create_registered_model(model_name)

        # Create a new model version and transition to 'Staging'
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run.info.run_id
        )
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='Staging'
        )

        print(f"Model registered with name: {model_name} and version: {model_version.version}")
        print(f"Model version {model_version.version} transitioned to 'Staging' stage")

        return model_name, model_version.version, X_test

def load_and_infer_model(model_name, model_version, input_data):
    # Load the model from MLflow
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Perform inference
    predictions = model.predict(input_data)
    
    return predictions

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=200, help="Maximum number of iterations for the Logistic Regression model")
    args = parser.parse_args()

    # Train, log the model and get model information
    model_name, model_version, X_test = train_and_log_model(args.max_iter)

    # Perform inference with the same data used for testing
    predictions = load_and_infer_model(model_name, model_version, X_test[:5])  # Sample for demonstration

    print("Sample Data for Inference:")
    print(X_test[:5])
    print("Predictions:")
    print(predictions)
