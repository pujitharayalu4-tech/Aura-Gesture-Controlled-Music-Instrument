"""
train_gesture_classifier.py
----------------------------
Trains a classifier on the landmark dataset produced by collect_gesture_data.py
and saves the trained model + label encoder for use in realtime_predict.py.

Usage:
    python train_gesture_classifier.py --data gesture_data.csv
"""

import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="gesture_data.csv")
    parser.add_argument("--model_out", default="gesture_model.pkl")
    parser.add_argument("--encoder_out", default="label_encoder.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} gesture classes:")
    print(df["label"].value_counts())

    X = df.drop(columns=["label"]).values
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # RandomForest is a strong default for small/medium tabular landmark data:
    # robust to feature scale differences, handles non-linear gesture boundaries,
    # and needs far less data/tuning than a neural net for this problem size.
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(clf, args.model_out)
    joblib.dump(encoder, args.encoder_out)
    print(f"\nSaved model to {args.model_out}")
    print(f"Saved label encoder to {args.encoder_out}")


if __name__ == "__main__":
    main()
