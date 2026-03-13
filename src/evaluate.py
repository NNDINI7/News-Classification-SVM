from sklearn.metrics import accuracy_score, confusion_matrix
from src.config import RESULT_PATH

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    cm = confusion_matrix(y_test, predictions)

    with open(RESULT_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))

    return accuracy