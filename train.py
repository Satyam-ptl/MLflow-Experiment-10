import sys
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=42
)

mlflow.set_experiment("Iris-Classification")

with mlflow.start_run() as run:
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

    print("Model trained successfully!")
    print("Number of estimators:", n_estimators)
    print("Accuracy:", accuracy)
    print("Run ID:", run.info.run_id)
