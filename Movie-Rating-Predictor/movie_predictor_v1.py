
# Movie Rating Predictor
# Random Forest Regression
# Individual and Batch CSV Prediction

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

TRAINING_FILE = BASE_DIR / "movies_v1_training.csv"
PRACTICE_RESULTS_FILE = BASE_DIR / "practice_test_results.csv"

NUMERIC_FEATURES = [
    "year",
    "runtime"
]

CATEGORICAL_FEATURES = [
    "genre",
    "director",
    "star",
    "writer",
    "rating",
    "country"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# CREATE MODEL
# --------------------------------------------------

def create_model():
    """Create and return the Random Forest pipeline."""

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=2
        ))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer,
         CATEGORICAL_FEATURES)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=40,
            random_state=42,
            n_jobs=1,
            min_samples_leaf=2
        ))
    ])

    return model


# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------

def load_training_data():
    """Load and prepare the training dataset."""

    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {TRAINING_FILE.name}."
        )

    data = pd.read_csv(TRAINING_FILE)

    required_columns = FEATURES + ["score", "name"]

    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Training CSV is missing columns: {missing_columns}"
        )

    # Remove rows without actual ratings
    data["score"] = pd.to_numeric(
        data["score"], errors="coerce"
    )

    data = data.dropna(subset=["score"]).copy()

    # Convert numerical features
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    return data


# --------------------------------------------------
# TRAIN AND EVALUATE MODEL
# --------------------------------------------------

def train_and_evaluate():
    """Evaluate the model, save results, and train final model."""

    print("\nLoading training dataset...")

    data = load_training_data()

    X = data[FEATURES]
    y = data["score"]

    print(f"Loaded {len(data)} movies.")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test, names_train, names_test = (
        train_test_split(
            X,
            y,
            data["name"],
            test_size=0.20,
            random_state=310
        )
    )

    print("\nTraining evaluation model...")

    evaluation_model = create_model()

    evaluation_model.fit(X_train, y_train)

    # Predict ratings for movies not used during training
    predictions = evaluation_model.predict(X_test)

    # Calculate Mean Absolute Error
    mae = mean_absolute_error(y_test, predictions)

    # Create test results
    results = pd.DataFrame({
        "name": names_test.values,
        "actual_score": y_test.values,
        "predicted_score": predictions
    })

    results["error"] = (
        results["actual_score"] -
        results["predicted_score"]
    ).abs()

    results = results.sort_values(
        by="error", ascending=False
    )

    # Save practice results
    results.to_csv(PRACTICE_RESULTS_FILE, index=False)

    print("\nEvaluation Complete!")
    print("------------------------------")
    print(f"Movies tested: {len(results)}")
    print(f"Mean Absolute Error: {mae:.3f}")

    within_one = (results["error"] <= 1.0).sum()

    print(
        f"Within +/- 1.0 point: "
        f"{within_one}/{len(results)} "
        f"({within_one / len(results) * 100:.1f}%)"
    )

    print(
        f"\nResults saved to: "
        f"{PRACTICE_RESULTS_FILE.name}"
    )

    # Train the final model on all available training data
    print("\nTraining final prediction model...")

    final_model = create_model()
    final_model.fit(X, y)

    print("Final model is ready!")

    return final_model


# --------------------------------------------------
# INDIVIDUAL MOVIE PREDICTION
# --------------------------------------------------

def predict_individual(model):
    """Collect movie information and predict its rating."""

    print("\nINDIVIDUAL MOVIE PREDICTION")
    print("------------------------------")

    movie = {}

    movie["genre"] = input("Genre: ")
    movie["year"] = float(input("Year: "))
    movie["rating"] = input("Rating (PG, PG-13, R, etc.): ")
    movie["director"] = input("Director: ")
    movie["writer"] = input("Writer: ")
    movie["star"] = input("Star: ")
    movie["country"] = input("Country: ")
    movie["runtime"] = float(input("Runtime in minutes: "))

    movie_data = pd.DataFrame([movie])

    prediction = model.predict(movie_data[FEATURES])[0]

    print("\nPrediction Results")
    print("------------------------------")
    print(f"Predicted Movie Score: {prediction:.2f}/10")


# --------------------------------------------------
# BATCH CSV PREDICTION
# --------------------------------------------------

def predict_batch(model):
    """Predict ratings for movies contained in a CSV file."""

    print("\nBATCH CSV PREDICTION")
    print("------------------------------")

    filename = input(
        "Enter the CSV filename or path: "
    ).strip()

    input_path = Path(filename).expanduser()

    # Check current directory, then script directory
    if not input_path.is_absolute() and not input_path.exists():
        input_path = BASE_DIR / input_path

    if not input_path.exists():
        print("Error: CSV file could not be found.")
        return

    try:
        batch_data = pd.read_csv(input_path)

        missing_columns = [
            column for column in FEATURES
            if column not in batch_data.columns
        ]

        if missing_columns:
            print(
                "Error: CSV is missing required columns: "
                f"{missing_columns}"
            )
            return

        # Convert numerical inputs
        for column in NUMERIC_FEATURES:
            batch_data[column] = pd.to_numeric(
                batch_data[column], errors="coerce"
            )

        print(f"\nLoaded {len(batch_data)} movies.")

        # Generate predictions
        predictions = model.predict(
            batch_data[FEATURES]
        )

        results = batch_data.copy()

        results["predicted_score"] = predictions

        # Evaluate predictions if actual scores are supplied
        if "score" in results.columns:

            results["actual_score"] = pd.to_numeric(
                results["score"], errors="coerce"
            )

            results["error"] = (
                results["actual_score"] -
                results["predicted_score"]
            ).abs()

            scored_results = results.dropna(
                subset=["actual_score"]
            )

            if not scored_results.empty:
                mae = mean_absolute_error(
                    scored_results["actual_score"],
                    scored_results["predicted_score"]
                )

                print("\nBatch Evaluation")
                print("------------------------------")
                print(
                    f"Movies evaluated: {len(scored_results)}"
                )
                print(f"Mean Absolute Error: {mae:.3f}")

        # Create output filename
        output_path = input_path.with_name(
            f"{input_path.stem}_results.csv"
        )

        results.to_csv(output_path, index=False)

        print("\nBatch prediction completed!")
        print(f"Movies processed: {len(results)}")
        print(f"Results saved to: {output_path}")

    except Exception as error:
        print(f"\nAn error occurred: {error}")


# --------------------------------------------------
# MAIN MENU
# --------------------------------------------------

def main():
    """Run the Movie Rating Predictor."""

    print("==============================")
    print("     MOVIE RATING PREDICTOR")
    print("==============================")

    try:
        model = train_and_evaluate()

    except Exception as error:
        print(f"\nUnable to start predictor: {error}")
        return

    while True:

        print("\nMAIN MENU")
        print("------------------------------")
        print("1. Predict an individual movie")
        print("2. Predict movies from a CSV file")
        print("3. Exit")

        choice = input("\nSelect an option: ").strip()

        if choice == "1":
            try:
                predict_individual(model)
            except ValueError:
                print("Please enter valid numerical values.")

        elif choice == "2":
            predict_batch(model)

        elif choice == "3":
            print("\nExiting Movie Rating Predictor.")
            break

        else:
            print("Invalid selection. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
