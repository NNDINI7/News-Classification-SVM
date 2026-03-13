from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.train import train_model
from src.evaluate import evaluate_model


def main():

    print("Step 1: Data Preprocessing")
    df = preprocess_data()

    print("Step 2: Feature Engineering")
    X, vectorizer = create_features(df["text"])

    y = df["category"]

    print("Step 3: Model Training")
    model, X_test, y_test = train_model(X, y)

    print("Step 4: Model Evaluation")
    accuracy = evaluate_model(model, X_test, y_test)

    print("Final Accuracy:", accuracy)


if __name__ == "__main__":
    main()