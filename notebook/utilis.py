from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator
import numpy as np

def mscore_to_int(x: str) -> int:
    """
    Converts MScore to integer class.
    0 = Low credit risk (A or B), 1 = High credit risk (C and above)
    
    Parameters:
        x (str): MScore value as a single-character string.

    Returns:
        int: 0 or 1 depending on credit risk level.
    """
    return 0 if str(x).strip().lower() in ['a', 'b'] else 1

def print_performances(
    name: str,
    classifier: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    digits: int = 2
) -> None:
    """
    Prints training and test accuracy, and a detailed classification report.

    Parameters:
        name (str): Name of the model.
        classifier (BaseEstimator): Trained scikit-learn classifier.
        X_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test feature set.
        y_test (np.ndarray): Test labels.
        digits (int): Number of decimal digits in classification report.
    """
    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train) * 100
    test_acc = accuracy_score(y_test, y_pred_test) * 100

    print(f"{name}")
    print(f" - Train Accuracy: {train_acc:.1f}%")
    print(f" - Test Accuracy : {test_acc:.1f}%\n")
    print("Test Classification Report:")
    print(classification_report(y_test, y_pred_test, digits=digits))
