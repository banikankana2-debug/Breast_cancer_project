from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras  # pyright: ignore[reportMissingModuleSource]


def main():
    keras.utils.set_random_seed(42)
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
        ],
        verbose=0,
    )
    predictions = (model.predict(X_test, verbose=0).ravel() >= 0.5).astype(int)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Train/test: {len(y_train)}/{len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("Confusion matrix (rows=true, columns=predicted; malignant, benign):")
    print(confusion_matrix(y_test, predictions))
    print("Classification report:")
    print(
        classification_report(
            y_test, predictions, target_names=data.target_names, digits=4
        )
    )


if __name__ == "__main__":
    main()
