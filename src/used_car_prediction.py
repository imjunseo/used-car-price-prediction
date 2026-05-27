"""
Used Vehicle Price Prediction Project

Main Techniques:
- Data Cleaning
- Missing Value Handling
- Feature Engineering
- Feature Scaling Comparison
- One-Hot Encoding
- Random Forest Regression
- Hyperparameter Tuning
- Cross Validation
- Price Gap Analysis
- Visualization
"""

# =============================================================
# Part 1 — Import Libraries
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    MinMaxScaler
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)


# =============================================================
# Part 2 — Main Function
# =============================================================

def run_vehicle_price_prediction(
    file_path,
    target_column="price"
):

    # =========================================================
    # Step 1: Load Dataset
    # =========================================================

    df = pd.read_csv(file_path)

    # =========================================================
    # Step 2: Missing Value Analysis
    # =========================================================

    missing_ratio = (
        df.isnull().sum() / len(df)
    ).sort_values(ascending=False)

    missing_ratio = missing_ratio[
        missing_ratio > 0
    ]

    if not missing_ratio.empty:

        plt.figure(figsize=(10, 6))

        missing_ratio.plot(kind="bar")

        plt.ylabel("Missing Ratio")
        plt.title("Missing Value Ratio by Feature")

        plt.tight_layout()

        plt.show()

    # =========================================================
    # Step 3: Data Cleaning
    # =========================================================

    # Remove invalid prices
    df = df[df[target_column] > 0]

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove unnecessary columns
    drop_columns = [
        "id",
        "url",
        "region_url",
        "VIN",
        "image_url",
        "description",
        "county",
        "posting_date",
        "model"
    ]

    df = df.drop(
        columns=drop_columns,
        errors="ignore"
    )

    # =========================================================
    # Step 4: Feature Engineering
    # =========================================================

    # Vehicle age
    current_year = 2025

    df["car_age"] = current_year - df["year"]

    # Mileage per year
    df["odometer_per_year"] = (
        df["odometer"] / (df["car_age"] + 1)
    )

    # High mileage indicator
    df["is_high_mileage"] = np.where(
        df["odometer"] > 150000,
        1,
        0
    )

    # =========================================================
    # Step 5: Remove Outliers
    # =========================================================

    outlier_columns = [
        "price",
        "year",
        "odometer",
        "car_age",
        "odometer_per_year"
    ]

    for col in outlier_columns:

        if col in df.columns:

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            df = df[
                (df[col] >= lower) &
                (df[col] <= upper)
            ]

    print(f"Cleaned Dataset Shape: {df.shape}")

    # =========================================================
    # Step 6: Sampling
    # =========================================================

    if len(df) > 10000:

        df = df.sample(
            n=10000,
            random_state=42
        )

    print(f"Sampled Dataset Shape: {df.shape}")

    # =========================================================
    # Step 7: Train / Test Split
    # =========================================================

    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]

    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]

    # =========================================================
    # Step 8: Feature Type Detection
    # =========================================================

    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    print(f"\nNumerical Features ({len(numerical_features)}):")
    print(numerical_features)

    print(f"\nCategorical Features ({len(categorical_features)}):")
    print(categorical_features)

    # =========================================================
    # Step 9: Rare Category Grouping
    # =========================================================

    for col in categorical_features:

        top_categories = (
            X_train[col]
            .value_counts()
            .index[:15]
        )

        X_train[col] = np.where(
            X_train[col].isin(top_categories),
            X_train[col],
            "Other"
        )

        X_test[col] = np.where(
            X_test[col].isin(top_categories),
            X_test[col],
            "Other"
        )

    # =========================================================
    # Step 10: Categorical Pipeline
    # =========================================================

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # =========================================================
    # Step 11: Scaler Comparison
    # =========================================================

    scalers = {
        "RobustScaler": RobustScaler(),
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler()
    }

    scaler_scores = {}

    print("\n=== Scaler Comparison (5-fold CV R²) ===")

    for scaler_name, scaler in scalers.items():

        numeric_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    scaler
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_transformer,
                    numerical_features
                ),
                (
                    "cat",
                    categorical_transformer,
                    categorical_features
                )
            ]
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=30,
                        max_depth=10,
                        random_state=42,
                        n_jobs=-1
                    )
                )
            ]
        )

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="r2",
            n_jobs=-1
        )

        mean_score = cv_scores.mean()

        scaler_scores[scaler_name] = mean_score

        print(
            f"{scaler_name:<15}: "
            f"CV R² = {mean_score:.4f}"
        )

    # =========================================================
    # Step 12: Select Best Scaler
    # =========================================================

    best_scaler_name = max(
        scaler_scores,
        key=scaler_scores.get
    )

    best_scaler = scalers[best_scaler_name]

    print(f"\nBest Scaler: {best_scaler_name}")

    # =========================================================
    # Plot: Scaler Comparison
    # =========================================================

    plt.figure(figsize=(7, 5))

    bars = plt.bar(
        scaler_scores.keys(),
        scaler_scores.values()
    )

    # Display score values above bars
    for bar, score in zip(
        bars,
        scaler_scores.values()
    ):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.002,
            f"{score:.4f}",
            ha="center"
        )

    plt.ylabel("Cross Validation R² Score")

    plt.title("Scaler Comparison")

    plt.ylim(0, 1)

    plt.tight_layout()

    plt.show()

    # =========================================================
    # Step 13: Final Pipeline
    # =========================================================

    best_numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                best_scaler
            )
        ]
    )

    best_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                best_numeric_transformer,
                numerical_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    final_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                best_preprocessor
            ),
            (
                "regressor",
                RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    # =========================================================
    # Step 14: Hyperparameter Tuning
    # =========================================================

    param_grid = {
        "regressor__n_estimators": [30, 50],
        "regressor__max_depth": [10, 15],
        "regressor__min_samples_split": [2],
        "regressor__min_samples_leaf": [1]
    }

    random_search = RandomizedSearchCV(
        estimator=final_pipeline,
        param_distributions=param_grid,
        n_iter=3,
        cv=5,
        scoring="r2",
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    print(f"\nBest CV R²: {random_search.best_score_:.4f}")

    print("\nBest Parameters:")
    print(random_search.best_params_)

    best_model = random_search.best_estimator_

    # =========================================================
    # Step 15: Prediction
    # =========================================================

    y_pred = best_model.predict(X_test)

    # =========================================================
    # Step 16: Evaluation Metrics
    # =========================================================

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    mape = mean_absolute_percentage_error(
        y_test,
        y_pred
    )

    print("\n====== Model Evaluation ======")

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape * 100:.2f}%")

    # =========================================================
    # Step 17: Price Gap Analysis
    # =========================================================

    result_df = pd.DataFrame({
        "Actual_Price": y_test,
        "Predicted_Price": y_pred
    })

    result_df["Price_Gap"] = (
        result_df["Actual_Price"]
        - result_df["Predicted_Price"]
    )

    result_df["Deal_Label"] = np.where(
        result_df["Price_Gap"] < 0,
        "Overpriced",
        "Underpriced"
    )

    print("\n====== Price Gap Analysis ======")

    print(
        result_df[
            [
                "Actual_Price",
                "Predicted_Price",
                "Price_Gap",
                "Deal_Label"
            ]
        ].head(10)
    )

    # =========================================================
    # Plot 1: Actual vs Predicted
    # =========================================================

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.5
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--"
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")

    plt.title("Actual vs Predicted Price")

    plt.tight_layout()

    plt.show()

    # =========================================================
    # Plot 2: Feature Importance
    # =========================================================

    feature_names = (
        best_model
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = (
        best_model
        .named_steps["regressor"]
        .feature_importances_
    )

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    ).head(20)

    plt.figure(figsize=(10, 8))

    plt.barh(
        feature_df["Feature"],
        feature_df["Importance"]
    )

    plt.xlabel("Importance")

    plt.title("Top 20 Feature Importances")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.show()

    # =========================================================
    # Plot 3: Price Distribution
    # =========================================================

    plt.figure(figsize=(8, 6))

    plt.hist(
        df[target_column],
        bins=50
    )

    plt.xlabel("Price")
    plt.ylabel("Frequency")

    plt.title("Vehicle Price Distribution")

    plt.tight_layout()

    plt.show()

    # =========================================================
    # Plot 4: Residual Plot
    # =========================================================

    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.5
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")

    plt.title("Residual Plot")

    plt.tight_layout()

    plt.show()

    # =========================================================
    # Plot 5: Correlation Heatmap
    # =========================================================

    numeric_df = df.select_dtypes(
        include=["int64", "float64"]
    )

    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.show()

    return best_model


# =============================================================
# Run Project
# =============================================================

if __name__ == "__main__":

    model = run_vehicle_price_prediction(
        file_path=r"C:\python\vehicles.csv"
    )
