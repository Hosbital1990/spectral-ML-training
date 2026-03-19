import pickle
import sys
from collections import Counter

import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MIN_SAMPLES_PER_CLASS = 3
INPUT_FILE = "ml_dataset_normalized.pkl"
EPOCHS = 20
BATCH_SIZE = 64


def main():
    try:
        import tensorflow as tf  # type: ignore[import-not-found]
    except ImportError:
        print("TensorFlow is required for this script.")
        print("Install it with: pip install tensorflow")
        sys.exit(1)

    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]

    print("Dataset loaded:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    label_counts = Counter(y)
    valid_labels = {
        label for label, count in label_counts.items() if count >= MIN_SAMPLES_PER_CLASS
    }

    mask = np.array([label in valid_labels for label in y])
    X_filtered = X[mask]
    y_filtered = y[mask]

    filtered_counts = Counter(y_filtered)

    print("\n--- CLASS FILTERING ---")
    print("Minimum samples per class required:", MIN_SAMPLES_PER_CLASS)
    print("Original samples:", len(y))
    print("Filtered samples:", len(y_filtered))
    print("Removed samples :", len(y) - len(y_filtered))
    print("Original classes:", len(label_counts))
    print("Remaining classes:", len(filtered_counts))

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_filtered)

    print("\nNumber of classes after filtering:", len(label_encoder.classes_))

    n_classes = len(label_encoder.classes_)
    n_samples = len(y_filtered)

    default_test_ratio = 0.2
    default_test_count = int(np.ceil(n_samples * default_test_ratio))
    safe_test_count = max(default_test_count, n_classes)

    if safe_test_count >= n_samples:
        raise ValueError(
            f"Not enough samples ({n_samples}) for stratified split with "
            f"{n_classes} classes. Increase filtering threshold."
        )

    test_size = safe_test_count / n_samples

    print("\n--- SPLIT INFO ---")
    print("Samples after filtering:", n_samples)
    print("Classes after filtering:", n_classes)
    print("Chosen test_size ratio:", round(test_size, 4))
    print("Chosen test sample count:", safe_test_count)

    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered,
        y_encoded,
        test_size=test_size,
        random_state=42,
        stratify=y_encoded,
    )

    print("\nTrain size:", X_train.shape)
    print("Test size :", X_test.shape)

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1], 1)),
            tf.keras.layers.Conv1D(32, kernel_size=5, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nTraining model...")
    model.fit(
        X_train,
        y_train_ohe,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    acc = accuracy_score(y_test, y_pred)

    print("\n--- RESULTS ---")
    print("Accuracy:", acc)

    print("\nClassification Report:")
    all_labels = np.arange(len(label_encoder.classes_))
    print(
        classification_report(
            y_test,
            y_pred,
            labels=all_labels,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
