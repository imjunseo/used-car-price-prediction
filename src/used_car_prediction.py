"""
Used Vehicle Price Prediction Project

Main Techniques:
- Data Cleaning
- Missing Value Handling
- Feature Engineering
- Scaling Method Comparison
- Encoding Method Comparison
- Multiple Regression Models
- Hyperparameter Tuning
- 5-Fold Cross Validation
- Price Gap Analysis
- Visualization
"""

# =============================================================
# Part 1 — Import Libraries
# =============================================================

import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# train_test_split:
# Splits the dataset into training and testing sets

# RandomizedSearchCV:
# Randomly searches hyperparameter combinations
# using cross-validation

# cross_val_score:
# Evaluates model performance using K-fold validation

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score
)

# ColumnTransformer:
# Applies different preprocessing methods
# to numerical and categorical features

from sklearn.compose import ColumnTransformer

# Pipeline:
# Sequentially combines preprocessing
# and machine learning steps into one workflow

from sklearn.pipeline import Pipeline

# SimpleImputer:
# Fills missing values using statistical methods

from sklearn.impute import SimpleImputer

# OneHotEncoder:
# Converts categorical variables into binary vectors

# OrdinalEncoder:
# Converts categories into integer labels

# RobustScaler:
# Scaling method robust to outliers

# StandardScaler:
# Standardizes features using mean and standard deviation

# MinMaxScaler:
# Scales features into a fixed range between 0 and 1

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    MinMaxScaler
)

# RandomForestRegressor:
# Ensemble learning model based on
# multiple decision trees

from sklearn.ensemble import RandomForestRegressor

# LinearRegression:
# Basic linear regression model

# Ridge:
# Linear regression model with L2 regularization

from sklearn.linear_model import (
    LinearRegression,
    Ridge
)

# MAE:
# Average absolute prediction error

# RMSE:
# Square root of mean squared error

# R²:
# Measures how well the model explains variance

# MAPE:
# Mean absolute percentage error

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
    # Create Output Folder
    # =========================================================

    os.makedirs("images", exist_ok=True)

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

        plt.title(
            "Missing Value Ratio by Feature"
        )

        plt.tight_layout()

        plt.savefig(
            "images/missing_value_ratio.png",
            dpi=300
        )

        plt.show()

    # =========================================================
    # Step 3: Data Cleaning
    # =========================================================

    df = df[df[target_column] > 0]

    df = df.drop_duplicates()

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

    current_year = 2025

    # Create vehicle age feature
    # because older vehicles generally
    # have lower prices

    df["car_age"] = (
        current_year - df["year"]
    )

    # Create mileage-per-year feature
    # to estimate annual vehicle usage

    df["odometer_per_year"] = (
        df["odometer"] /
        (df["car_age"] + 1)
    )

    # Create high mileage indicator

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

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[
            (df[col] >= lower) &
            (df[col] <= upper)
        ]

    # =========================================================
    # Step 6: Sampling
    # =========================================================

    if len(df) > 10000:

        df = df.sample(
            n=10000,
            random_state=42
        )

    # =========================================================
    # Step 7: Train/Test Split
    # =========================================================

    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    X_train = df_train.drop(
        columns=[target_column]
    )

    y_train = df_train[target_column]

    X_test = df_test.drop(
        columns=[target_column]
    )

    y_test = df_test[target_column]

    # =========================================================
    # Step 8: Feature Type Detection
    # =========================================================

    numerical_features = (
        X_train.select_dtypes(
            include=["int64", "float64"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X_train.select_dtypes(
            include=["object", "string"]
        )
        .columns
        .tolist()
    )

    # =========================================================
    # Step 9: Rare Category Grouping
    # =========================================================

    # Rare categories are grouped into "Other"
    # to reduce dimensionality after encoding

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
    # Step 10: Scalers and Encoders
    # =========================================================

    scalers = {
        "RobustScaler": RobustScaler(),
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler()
    }

    encoders = {
        "OneHotEncoder": OneHotEncoder(
            handle_unknown="ignore"
        ),

        "OrdinalEncoder": OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    }

    # =========================================================
    # Step 11: Model Definitions
    # =========================================================

    # Multiple regression models are compared
    # to identify the best overall model

    models = {

        "LinearRegression":
            LinearRegression(),

        "Ridge":
            Ridge(),

        "RandomForest":
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
    }

    # =========================================================
    # Step 12: Parameter Grids
    # =========================================================

    # Hyperparameter combinations
    # for each regression model

    param_grids = {

        "LinearRegression": {},

        "Ridge": {
            "regressor__alpha":
                [0.1, 1.0, 10.0]
        },

        "RandomForest": {

            "regressor__n_estimators":
                [30, 50, 100],

            "regressor__max_depth":
                [10, 15, 20],

            "regressor__min_samples_split":
                [2, 5],

            "regressor__min_samples_leaf":
                [1, 2]
        }
    }

    # =========================================================
    # Step 13: Find Best Preprocessing Combination
    # =========================================================

    preprocessing_results = []

    best_preprocessor = None
    best_scaler_name = None
    best_encoder_name = None

    best_preprocess_score = -np.inf

    for model_name, model in models.items():

        print(
            f"\n========== "
            f"{model_name} Preprocessing Comparison =========="
        )

        for scaler_name, scaler in scalers.items():

            for encoder_name, encoder in encoders.items():

                # Numerical preprocessing pipeline

                numeric_transformer = Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        ),
                        (
                            "scaler",
                            scaler
                        )
                    ]
                )

                # Categorical preprocessing pipeline

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
                            encoder
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
                            model
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

                preprocessing_results.append({

                    "Model":
                        model_name,

                    "Scaler":
                        scaler_name,

                    "Encoder":
                        encoder_name,

                    "CV_R2":
                        mean_score
                })

                print(
                    f"{scaler_name:<15} "
                    f"{encoder_name:<18} "
                    f"CV R² = {mean_score:.4f}"
                )

                if mean_score > best_preprocess_score:

                    best_preprocess_score = mean_score

                    best_scaler_name = scaler_name
                    best_encoder_name = encoder_name

                    best_preprocessor = preprocessor

    preprocessing_df = pd.DataFrame(
        preprocessing_results
    )

    preprocessing_df.to_csv(
        "images/preprocessing_leaderboard.csv",
        index=False
    )

    print(
        f"\nBest Scaler: {best_scaler_name}"
    )

    print(
        f"Best Encoder: {best_encoder_name}"
    )

    # =========================================================
    # Plot 1: Preprocessing Comparison
    # =========================================================

    plt.figure(figsize=(12, 6))

    labels = (
        preprocessing_df["Model"]
        + "\n"
        + preprocessing_df["Scaler"]
        + "\n"
        + preprocessing_df["Encoder"]
    )

    plt.bar(
        labels,
        preprocessing_df["CV_R2"]
    )

    plt.xticks(rotation=90)

    plt.ylabel("CV R² Score")

    plt.title(
        "Preprocessing Combination Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        "images/preprocessing_comparison.png",
        dpi=300
    )

    plt.show()

    # =========================================================
    # Step 14: Model Training and Comparison
    # =========================================================

    results = []

    best_model = None
    best_model_name = None

    best_score = -np.inf

    for model_name, model in models.items():

        print(
            f"\n========== "
            f"{model_name} =========="
        )

        final_pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    best_preprocessor
                ),
                (
                    "regressor",
                    model
                )
            ]
        )

        param_grid = param_grids[
            model_name
        ]

        if param_grid:

            # RandomizedSearchCV searches random
            # hyperparameter combinations
            # using cross-validation

            search = RandomizedSearchCV(
                estimator=final_pipeline,
                param_distributions=param_grid,
                n_iter=5,
                cv=5,
                scoring="r2",
                random_state=42,
                n_jobs=-1
            )

        else:

            search = final_pipeline

        # =====================================================
        # Train Model
        # =====================================================

        if param_grid:

            search.fit(
                X_train,
                y_train
            )

            current_model = (
                search.best_estimator_
            )

            cv_score = (
                search.best_score_
            )

            best_params = (
                search.best_params_
            )

        else:

            search.fit(
                X_train,
                y_train
            )

            current_model = search

            cv_scores = cross_val_score(
                current_model,
                X_train,
                y_train,
                cv=5,
                scoring="r2",
                n_jobs=-1
            )

            cv_score = cv_scores.mean()

            best_params = {}

        print("\nBest Parameters:")

        print(best_params)

        # =====================================================
        # Prediction
        # =====================================================

        y_pred = current_model.predict(
            X_test
        )

        # =====================================================
        # Evaluation
        # =====================================================

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

        mape = (
            mean_absolute_percentage_error(
                y_test,
                y_pred
            ) * 100
        )

        print(
            f"CV R² : "
            f"{cv_score:.4f}"
        )

        print(f"MAE   : {mae:.2f}")

        print(f"RMSE  : {rmse:.2f}")

        print(f"R²    : {r2:.4f}")

        print(f"MAPE  : {mape:.2f}%")

        results.append({

            "Model":
                model_name,

            "CV_R2":
                cv_score,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "R2":
                r2,

            "MAPE":
                mape
        })

        # =====================================================
        # Select Best Model
        # =====================================================

        if r2 > best_score:

            best_score = r2

            best_model = current_model

            best_model_name = model_name

    # =========================================================
    # Step 15: Leaderboard
    # =========================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="R2",
        ascending=False
    )

    print(
        "\n====== Model Leaderboard ======"
    )

    print(results_df)

    results_df.to_csv(
        "images/model_leaderboard.csv",
        index=False
    )

    print(
        f"\nBest Overall Model: "
        f"{best_model_name}"
    )

    # =========================================================
    # Final Prediction Using Best Model
    # =========================================================

    y_pred = best_model.predict(X_test)

    # =========================================================
    # Step 16: Price Gap Analysis
    # =========================================================

    result_df = pd.DataFrame({

        "Actual_Price":
            y_test,

        "Predicted_Price":
            y_pred
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

    print(
        "\n====== Price Gap Analysis ======"
    )

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
    # Plot 2: Actual vs Predicted
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

    plt.title(
        "Actual vs Predicted Price"
    )

    plt.tight_layout()

    plt.savefig(
        "images/actual_vs_predicted.png",
        dpi=300
    )

    plt.show()

    # =========================================================
    # Plot 3: Feature Importance
    # =========================================================

    regressor = best_model.named_steps[
        "regressor"
    ]

    # Tree-based models use feature_importances_
    # Linear models use coefficient magnitude

    if hasattr(
        regressor,
        "feature_importances_"
    ):

        importances = (
            regressor.feature_importances_
        )

    elif hasattr(
        regressor,
        "coef_"
    ):

        importances = np.abs(
            regressor.coef_
        )

    else:

        importances = None

    if importances is not None:

        feature_names = (

            best_model
            .named_steps["preprocessor"]
            .get_feature_names_out()
        )

        feature_df = pd.DataFrame({

            "Feature":
                feature_names,

            "Importance":
                importances
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

        plt.title(
            "Top 20 Feature Importances"
        )

        plt.gca().invert_yaxis()

        plt.tight_layout()

        plt.savefig(
            "images/feature_importance.png",
            dpi=300
        )

        plt.show()

    # =========================================================
    # Plot 4: Residual Plot
    # =========================================================

    # Residual plot visualizes prediction errors

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

    plt.savefig(
        "images/residual_plot.png",
        dpi=300
    )

    plt.show()

    # =========================================================
    # Plot 5: Correlation Heatmap
    # =========================================================

    # Correlation heatmap visualizes
    # relationships between numerical features

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

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        "images/correlation_heatmap.png",
        dpi=300
    )

    plt.show()

    return best_model


# =============================================================
# Run Project
# =============================================================

if __name__ == "__main__":

    model = run_vehicle_price_prediction(
        file_path=r"C:\python\vehicles.csv"
    )