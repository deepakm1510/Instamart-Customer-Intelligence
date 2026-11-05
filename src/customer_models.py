
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# FILE LOCATIONS
# ==========================================

DATA_DIR = Path("data/clean")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD CLEAN DATA
# ==========================================

print("\n========== LOADING DATA ==========")

customers = pd.read_csv(DATA_DIR / "customers.csv")
products = pd.read_csv(DATA_DIR / "products.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))


# ==========================================
# PART B1 - RFM CALCULATION
# ==========================================

print("\n========== RFM CALCULATION ==========")

completed = orders[orders["status"] == "completed"].copy()

snapshot_date = pd.Timestamp("2025-10-01 00:00:00")

rfm = completed.groupby("customer_id").agg(
    last_order_date=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

rfm["recency"] = (
    snapshot_date - rfm["last_order_date"]
).dt.days

rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

rfm.to_csv(OUTPUT_DIR / "customer_rfm.csv", index=False)

print("RFM customers:", len(rfm))
print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

print("Saved: outputs/customer_rfm.csv")


# ==========================================
# PART B2 - K-MEANS EVALUATION
# ==========================================

print("\n========== K-MEANS EVALUATION ==========")

rfm_features = rfm[["recency", "frequency", "monetary"]]

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

clustering_results = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(rfm_scaled)

    clustering_results.append({
        "K": k,
        "Inertia": round(kmeans.inertia_, 2),
        "Silhouette": round(silhouette_score(rfm_scaled, labels), 3)
    })

clustering_df = pd.DataFrame(clustering_results)

print(clustering_df.to_string(index=False))

clustering_df.to_csv(
    OUTPUT_DIR / "clustering_evaluation.csv",
    index=False
)

print("Saved: outputs/clustering_evaluation.csv")


# ==========================================
# PART B3 - FINAL CUSTOMER SEGMENTS
# ==========================================

print("\n========== CUSTOMER SEGMENTATION ==========")

final_kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = final_kmeans.fit_predict(rfm_scaled)

segment_profile = rfm.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).round(2)

print(segment_profile)

# Cluster IDs are arbitrary.
# Assign business names using cluster profiles.

premium_segment = segment_profile["mean_monetary"].idxmax()
dormant_segment = segment_profile["mean_recency"].idxmax()

remaining_segments = [
    segment_id
    for segment_id in segment_profile.index
    if segment_id not in [premium_segment, dormant_segment]
]

loyal_segment = segment_profile.loc[
    remaining_segments, "mean_frequency"
].idxmax()

occasional_segment = [
    segment_id
    for segment_id in remaining_segments
    if segment_id != loyal_segment
][0]

segment_names = {
    premium_segment: "Premium",
    dormant_segment: "Dormant",
    loyal_segment: "Loyal",
    occasional_segment: "Occasional"
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)

rfm["is_premium"] = (
    rfm["segment_id"] == premium_segment
).astype(int)

print("\nSegment names:")
print(segment_names)

print("\nPremium segment ID:", premium_segment)

# PCA

pca = PCA(n_components=2)

pca_result = pca.fit_transform(rfm_scaled)

print("\n========== PCA EXPLAINED VARIANCE ==========")
print("PC1:", round(pca.explained_variance_ratio_[0], 3))
print("PC2:", round(pca.explained_variance_ratio_[1], 3))
print("Total:", round(pca.explained_variance_ratio_.sum(), 3))

segments_output = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
]

segments_output.to_csv(
    OUTPUT_DIR / "customer_segments.csv",
    index=False
)

print("Saved: outputs/customer_segments.csv")


# ==========================================
# PART B4 - LAPSE PREDICTION DATASET
# ==========================================

print("\n========== LAPSE PREDICTION DATASET ==========")

cutoff_date = pd.Timestamp("2025-09-01 00:00:00")
window_end = pd.Timestamp("2025-10-01 00:00:00")

# Add product category

orders = orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

# Features use only orders before cutoff date.

pre_cutoff = orders[
    orders["order_ts"] < cutoff_date
].copy()

pre_completed = pre_cutoff[
    pre_cutoff["status"] == "completed"
].copy()

pre_cancelled = pre_cutoff[
    pre_cutoff["status"] == "cancelled"
].copy()

# Cohort: customers with at least one completed
# order before cutoff date.

features = pre_completed.groupby("customer_id").agg(
    last_order_date=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    average_discount=("discount_pct", "mean"),
    average_delivery=("delivery_minutes", "mean")
).reset_index()

features["recency"] = (
    cutoff_date - features["last_order_date"]
).dt.days

# Cancellation rate

final_orders = pre_cutoff[
    pre_cutoff["status"].isin(["completed", "cancelled"])
]

total_final = final_orders.groupby("customer_id").size()
cancelled_counts = pre_cancelled.groupby("customer_id").size()

features["cancellation_rate"] = (
    cancelled_counts.reindex(features["customer_id"])
    .fillna(0).to_numpy()
    /
    total_final.reindex(features["customer_id"]).to_numpy()
)

# Favourite category

category_counts = (
    pre_completed.groupby(["customer_id", "category"])
    .size()
    .reset_index(name="category_count")
    .sort_values(
        ["customer_id", "category_count", "category"],
        ascending=[True, False, True]
    )
)

favourite_category = (
    category_counts.drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favourite_category"})
)

features = features.merge(
    favourite_category,
    on="customer_id",
    how="left"
)

# Customer profile

features = features.merge(
    customers[
        ["customer_id", "city", "age", "gender", "signup_date"]
    ],
    on="customer_id",
    how="left"
)

features["tenure_days"] = (
    cutoff_date - features["signup_date"]
).dt.days

# Label: no completed order during September.

september_completed = orders[
    (orders["status"] == "completed")
    & (orders["order_ts"] >= cutoff_date)
    & (orders["order_ts"] < window_end)
]

active_in_september = set(
    september_completed["customer_id"]
)

features["lapsed"] = (
    ~features["customer_id"].isin(active_in_september)
).astype(int)

features = features.drop(
    columns=["last_order_date", "signup_date"]
)

print("Cohort customers:", len(features))
print("Lapsed customers:", int(features["lapsed"].sum()))

print(
    "Lapse percentage:",
    round(features["lapsed"].mean() * 100, 2)
)

print("\nMissing feature values:")
print(features.isnull().sum())

features.to_csv(
    OUTPUT_DIR / "lapse_features.csv",
    index=False
)

print("Saved: outputs/lapse_features.csv")


# ==========================================
# PART B5 - TRAIN TEST SPLIT AND BASELINE
# ==========================================

print("\n========== TRAIN-TEST SPLIT ==========")

lapse_data = pd.read_csv(
    OUTPUT_DIR / "lapse_features.csv"
)

customer_ids = lapse_data["customer_id"]

X = lapse_data.drop(
    columns=["customer_id", "lapsed"]
)

y = lapse_data["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining class counts:")
print(y_train.value_counts())

print("\nTesting class counts:")
print(y_test.value_counts())

baseline_predictions = np.zeros(
    len(y_test),
    dtype=int
)

print("\n========== BASELINE MODEL ==========")

print(
    "Baseline accuracy:",
    round(accuracy_score(y_test, baseline_predictions), 4)
)

print(
    "Baseline F1:",
    round(f1_score(y_test, baseline_predictions, zero_division=0), 4)
)


# ==========================================
# PART B6 - PREPROCESSING AND MODEL TUNING
# ==========================================

print("\n========== MODEL TUNING ==========")

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "average_discount",
    "cancellation_rate",
    "average_delivery",
    "age",
    "tenure_days"
]

categorical_features = [
    "favourite_category",
    "city",
    "gender"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# Logistic Regression pipeline

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
)

logistic_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

logistic_search = GridSearchCV(
    logistic_pipeline,
    logistic_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("Training Logistic Regression...")

logistic_search.fit(X_train, y_train)

print("Best Logistic Parameters:", logistic_search.best_params_)
print("Best Logistic CV F1:", round(logistic_search.best_score_, 4))


# Random Forest pipeline

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

rf_grid = {
    "classifier__max_depth": [3, 5, 8],
    "classifier__min_samples_leaf": [5, 10, 20]
}

rf_search = GridSearchCV(
    rf_pipeline,
    rf_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTraining Random Forest...")

rf_search.fit(X_train, y_train)

print("Best Random Forest Parameters:", rf_search.best_params_)
print("Best Random Forest CV F1:", round(rf_search.best_score_, 4))

print("\nModel tuning completed successfully!")


# ==========================================
# PART B7 - TEST SET EVALUATION
# ==========================================

print("\n========== TEST SET EVALUATION ==========")

# Test set is used only for final evaluation.
# It will not be used to select the final model.

models = {
    "Logistic Regression": logistic_search.best_estimator_,
    "Random Forest": rf_search.best_estimator_
}

test_results = {}

for model_name, model in models.items():

    predictions = model.predict(X_test)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    test_results[model_name] = {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print("\n", model_name)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))


# ==========================================
# PART B8 - TRAINING-ONLY OVERSAMPLING
# ==========================================

print("\n========== OVERSAMPLING ==========")

# Oversampling is performed only on training data.
# Test data remains untouched.

X_lapsed = X_train[y_train == 1]
X_not_lapsed = X_train[y_train == 0]

y_lapsed = y_train[y_train == 1]
y_not_lapsed = y_train[y_train == 0]

X_lapsed_extra = resample(
    X_lapsed,
    replace=True,
    n_samples=len(X_not_lapsed) - len(X_lapsed),
    random_state=42
)

y_lapsed_extra = resample(
    y_lapsed,
    replace=True,
    n_samples=len(y_not_lapsed) - len(y_lapsed),
    random_state=42
)

X_train_balanced = pd.concat(
    [X_train, X_lapsed_extra]
)

y_train_balanced = pd.concat(
    [y_train, y_lapsed_extra]
)

print("Balanced training class counts:")
print(y_train_balanced.value_counts())

# Use the Logistic Regression C selected by
# cross-validation on the original training data.

oversampled_logistic = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                C=logistic_search.best_params_["classifier__C"]
            )
        )
    ]
)

oversampled_logistic.fit(
    X_train_balanced,
    y_train_balanced
)

oversampled_predictions = oversampled_logistic.predict(X_test)

oversampled_precision = precision_score(
    y_test,
    oversampled_predictions,
    zero_division=0
)

oversampled_recall = recall_score(
    y_test,
    oversampled_predictions,
    zero_division=0
)

oversampled_f1 = f1_score(
    y_test,
    oversampled_predictions,
    zero_division=0
)

print("\nOversampled Logistic Regression")
print("Confusion Matrix:")
print(confusion_matrix(y_test, oversampled_predictions))
print("Precision:", round(oversampled_precision, 4))
print("Recall:", round(oversampled_recall, 4))
print("F1:", round(oversampled_f1, 4))


# ==========================================
# PART B9 - SCALER LEAKAGE CHECK
# ==========================================

print("\n========== SCALER MEAN CHECK ==========")

fitted_preprocessor = (
    logistic_search.best_estimator_
    .named_steps["preprocessor"]
)

frequency_position = numeric_features.index("frequency")

training_frequency_mean = (
    fitted_preprocessor
    .named_transformers_["num"]
    .mean_[frequency_position]
)

cohort_frequency_mean = X["frequency"].mean()

print(
    "Training frequency mean:",
    round(training_frequency_mean, 4)
)

print(
    "Full cohort frequency mean:",
    round(cohort_frequency_mean, 4)
)

print(
    "Scaler was fitted only on training data, avoiding test-data leakage."
)


# ==========================================
# PART B10 - RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

print("\n========== RANDOM FOREST TOP FEATURES ==========")

best_rf = rf_search.best_estimator_

rf_preprocessor = best_rf.named_steps["preprocessor"]

rf_feature_names = rf_preprocessor.get_feature_names_out()

rf_importances = best_rf.named_steps[
    "classifier"
].feature_importances_

rf_importance_df = pd.DataFrame({
    "feature": rf_feature_names,
    "importance": rf_importances
}).sort_values(
    "importance",
    ascending=False
)

print(rf_importance_df.head(5).to_string(index=False))


# ==========================================
# PART B11 - LOGISTIC REGRESSION COEFFICIENTS
# ==========================================

print("\n========== LOGISTIC REGRESSION COEFFICIENTS ==========")

logistic_preprocessor = (
    oversampled_logistic.named_steps["preprocessor"]
)

logistic_feature_names = (
    logistic_preprocessor.get_feature_names_out()
)

logistic_coefficients = (
    oversampled_logistic.named_steps["classifier"].coef_[0]
)

coefficient_df = pd.DataFrame({
    "feature": logistic_feature_names,
    "coefficient": logistic_coefficients
})

coefficient_df["absolute_coefficient"] = (
    coefficient_df["coefficient"].abs()
)

coefficient_df = coefficient_df.sort_values(
    "absolute_coefficient",
    ascending=False
)

print(
    coefficient_df[
        ["feature", "coefficient"]
    ].head(3).to_string(index=False)
)


# ==========================================
# PART B12 - SELECT AND SAVE FINAL MODEL
# ==========================================

print("\n========== FINAL MODEL SELECTION ==========")

# PRE-DECLARED BUSINESS SELECTION:
#
# Missing a customer who is likely to lapse can result
# in lost future revenue.
#
# Therefore, we prioritise recall and select the
# training-only oversampled Logistic Regression model.
#
# IMPORTANT:
# Do NOT compare test-set F1 scores to choose a model.
# The test set is used only for final evaluation.

selected_model_name = "Oversampled Logistic Regression"

final_model = oversampled_logistic

print("Selected model:", selected_model_name)

print(
    "Selection reason: Recall is prioritised to identify more customers at risk of lapsing."
)

print("\nFinal test-set evaluation:")

print(
    "Precision:",
    round(oversampled_precision, 4)
)

print(
    "Recall:",
    round(oversampled_recall, 4)
)

print(
    "F1:",
    round(oversampled_f1, 4)
)

model_path = OUTPUT_DIR / "lapse_model.joblib"

joblib.dump(final_model, model_path)

print("Saved model:", model_path)

# Reload model and verify predictions.

loaded_model = joblib.load(model_path)

original_predictions = final_model.predict(X_test)
reloaded_predictions = loaded_model.predict(X_test)

print(
    "Reload prediction verification:",
    np.array_equal(original_predictions, reloaded_predictions)
)


# ==========================================
# PART B13 - PREMIUM RETENTION LIST
# ==========================================

print("\n========== PREMIUM RETENTION LIST ==========")

segments = pd.read_csv(
    OUTPUT_DIR / "customer_segments.csv"
)

premium_ids = set(
    segments.loc[
        segments["is_premium"] == 1,
        "customer_id"
    ]
)

# Get customer IDs corresponding to test rows.

test_customer_ids = customer_ids.loc[X_test.index]

premium_test_mask = test_customer_ids.isin(premium_ids)

premium_test_X = X_test.loc[premium_test_mask]

premium_test_ids = test_customer_ids.loc[
    premium_test_mask
]

if len(premium_test_X) > 0:

    premium_probabilities = loaded_model.predict_proba(
        premium_test_X
    )[:, 1]

    retention_list = pd.DataFrame({
        "customer_id": premium_test_ids.to_numpy(),
        "lapse_probability": premium_probabilities,
        "actually_lapsed": y_test.loc[
            premium_test_X.index
        ].to_numpy()
    })

    retention_list = retention_list.sort_values(
        "lapse_probability",
        ascending=False
    ).head(10)

else:

    retention_list = pd.DataFrame(
        columns=[
            "customer_id",
            "lapse_probability",
            "actually_lapsed"
        ]
    )

retention_list.to_csv(
    OUTPUT_DIR / "retention_list.csv",
    index=False
)

print("Premium customers in test set:", int(premium_test_mask.sum()))
print("Retention list rows:", len(retention_list))

if len(retention_list) > 0:
    print(
        "Actually lapsed in top 10:",
        int(retention_list["actually_lapsed"].sum())
    )

print("\nRetention list:")
print(retention_list.to_string(index=False))

print("\nSaved: outputs/retention_list.csv")


# ==========================================
# PART B COMPLETE
# ==========================================

print("\n====================================")
print("PART B COMPLETED SUCCESSFULLY!")
print("====================================")
