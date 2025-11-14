import mlflow
import pickle
import dagshub

dagshub.init(repo_owner='ankita.datta24', repo_name='MLFLOW-Tutorial', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/ankita.datta24/MLFLOW-Tutorial.mlflow")
mlflow.set_registry_uri(None)

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

max_depth = 8
n_estimators = 5

mlflow.set_experiment("YT-MLOPS-Exp2")

with mlflow.start_run():

    # Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log params and metrics
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)

    # Save model manually with pickle
    with open("rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)

    # Log model file as artifact
    mlflow.log_artifact("rf_model.pkl")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

    print("Accuracy:", accuracy)
