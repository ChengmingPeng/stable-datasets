import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stable_datasets.images import CitrusLeaves, TrashNet


def extract_features(dataset):
    images = []
    labels = []

    for sample in dataset:
        image = sample["image"].convert("RGB")
        image = image.resize((16, 16))

        features = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0

        images.append(features)
        labels.append(int(sample["label"]))

    return np.stack(images), np.asarray(labels)


def evaluate(name, X_train, X_test, y_train, y_test):
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            tol=1e-3,
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced = balanced_accuracy_score(y_test, predictions)

    print(f"\n{name}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced:.4f}")


def run_citrus():
    dataset = CitrusLeaves(split="train")
    X, y = extract_features(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    evaluate(
        "Citrus Leaves",
        X_train,
        X_test,
        y_train,
        y_test,
    )


def run_trashnet():
    train = TrashNet(split="train")
    test = TrashNet(split="test")

    X_train, y_train = extract_features(train)
    X_test, y_test = extract_features(test)

    evaluate(
        "TrashNet",
        X_train,
        X_test,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    run_citrus()
    run_trashnet()
