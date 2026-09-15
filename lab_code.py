"""
Lab 06 - 23CSE301

Requirements:
A1 - Repeat Lab 05 experiments using GenAI
A2 - Unit and functional testing
A3 - Performance comparison:
     1. Custom kNN
     2. Scikit-learn kNN
     3. GenAI kNN
"""

import random
import time
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ===============================================================
# A1 - ENCODING
# GenAI tool used: ChatGPT
# ===============================================================

def label_encode(series):
    categories = sorted(series.dropna().unique())
    mapping = {cat: i for i, cat in enumerate(categories)}
    return series.map(mapping), mapping


def one_hot_encode(df, column):
    return pd.get_dummies(df, columns=[column])


# ===============================================================
# A1 - DATA IMPUTATION
# GenAI tool used: ChatGPT
# ===============================================================

def impute_column(series, method="mean"):

    if method == "mean":
        return series.fillna(series.mean())

    elif method == "median":
        return series.fillna(series.median())

    elif method == "mode":
        return series.fillna(series.mode()[0])

    raise ValueError("Unknown method")


# ===============================================================
# A1 - DISTANCE FUNCTIONS
# GenAI tool used: ChatGPT
# ===============================================================

def euclidean_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return float(
        np.sqrt(np.sum((vec1 - vec2) ** 2))
    )


def manhattan_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return float(
        np.sum(np.abs(vec1 - vec2))
    )


DISTANCE_FUNCS = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance
}


# ===============================================================
# A1 - SORTING ALGORITHMS
# GenAI tool used: ChatGPT
# ===============================================================

def bubble_sort(items, key=lambda x: x):

    items = items[:]
    n = len(items)

    for i in range(n):

        for j in range(n - i - 1):

            if key(items[j]) > key(items[j + 1]):

                items[j], items[j + 1] = (
                    items[j + 1],
                    items[j]
                )

    return items


def merge_sort(items, key=lambda x: x):

    if len(items) <= 1:
        return items

    mid = len(items) // 2

    left = merge_sort(items[:mid], key)
    right = merge_sort(items[mid:], key)

    merged = []
    i = 0
    j = 0

    while i < len(left) and j < len(right):

        if key(left[i]) <= key(right[j]):
            merged.append(left[i])
            i += 1

        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged


def quick_sort(items, key=lambda x: x):

    if len(items) <= 1:
        return items

    pivot = items[len(items) // 2]

    less = [
        x for x in items
        if key(x) < key(pivot)
    ]

    equal = [
        x for x in items
        if key(x) == key(pivot)
    ]

    greater = [
        x for x in items
        if key(x) > key(pivot)
    ]

    return (
        quick_sort(less, key)
        + equal
        + quick_sort(greater, key)
    )


SORT_FUNCS = {
    "bubble": bubble_sort,
    "merge": merge_sort,
    "quick": quick_sort
}


# ===============================================================
# A1 - FIND K NEAREST NEIGHBORS
# GenAI tool used: ChatGPT
# ===============================================================

def get_k_neighbors(
    train_X,
    train_y,
    test_point,
    k,
    distance_metric="euclidean",
    sort_algo="quick"
):

    dist_func = DISTANCE_FUNCS[distance_metric]

    distances = []

    for i in range(len(train_X)):

        distance = dist_func(
            test_point,
            train_X[i]
        )

        distances.append(
            (distance, train_y[i], i)
        )

    sort_func = SORT_FUNCS[sort_algo]

    sorted_distances = sort_func(
        distances,
        key=lambda x: (x[0], x[2])
    )

    return sorted_distances[:k]


# ===============================================================
# A1 - MAJORITY VOTING
# GenAI tool used: ChatGPT
# ===============================================================

def majority_vote(neighbors):

    labels = [
        neighbor[1]
        for neighbor in neighbors
    ]

    counts = {}

    for label in labels:

        counts[label] = (
            counts.get(label, 0) + 1
        )

    max_count = max(counts.values())

    tied_labels = [
        label
        for label, count in counts.items()
        if count == max_count
    ]

    if len(tied_labels) == 1:
        return tied_labels[0]

    # Nearest neighbor wins in case of tie
    for distance, label, index in neighbors:

        if label in tied_labels:
            return label

    return tied_labels[0]


# ===============================================================
# A1 - WEIGHTED VOTING
# GenAI tool used: ChatGPT
# ===============================================================

def weighted_vote(neighbors, epsilon=1e-5):

    weights = {}

    for distance, label, index in neighbors:

        weight = 1 / (distance + epsilon)

        weights[label] = (
            weights.get(label, 0) + weight
        )

    return max(
        weights,
        key=weights.get
    )


# ===============================================================
# A1 - CUSTOM KNN
# GenAI tool used: ChatGPT
# ===============================================================

class CustomKNN:

    def __init__(
        self,
        k=3,
        distance_metric="euclidean",
        sort_algo="quick",
        weighted=False
    ):

        self.k = k
        self.distance_metric = distance_metric
        self.sort_algo = sort_algo
        self.weighted = weighted

        self.X_train = None
        self.y_train = None


    def fit(self, X_train, y_train):

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

        return self


    def _predict_one(self, x):

        neighbors = get_k_neighbors(
            self.X_train,
            self.y_train,
            x,
            self.k,
            self.distance_metric,
            self.sort_algo
        )

        if self.weighted:
            return weighted_vote(neighbors)

        return majority_vote(neighbors)


    def predict(self, X_test):

        X_test = np.array(X_test)

        return [
            self._predict_one(x)
            for x in X_test
        ]


    def score(self, X_test, y_test):

        predictions = self.predict(X_test)

        return accuracy_score(
            y_test,
            predictions
        )


# ===============================================================
# A1 - GENAI KNN
#
# This is the second independently implemented kNN version
# used for the A3 comparison.
#
# GenAI tool used: ChatGPT
# ===============================================================

class GenAIKNN:

    def __init__(self, k=3):

        self.k = k
        self.X_train = None
        self.y_train = None


    def fit(self, X, y):

        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

        return self


    def euclidean_distance(self, x1, x2):

        return np.sqrt(
            np.sum((x1 - x2) ** 2)
        )


    def predict(self, X):

        predictions = []

        for test_point in np.asarray(X):

            distances = []

            for i, train_point in enumerate(
                self.X_train
            ):

                distance = self.euclidean_distance(
                    test_point,
                    train_point
                )

                distances.append(
                    (distance, self.y_train[i])
                )

            distances.sort(
                key=lambda x: x[0]
            )

            nearest = distances[:self.k]

            labels = [
                label
                for distance, label in nearest
            ]

            values, counts = np.unique(
                labels,
                return_counts=True
            )

            prediction = values[
                np.argmax(counts)
            ]

            predictions.append(prediction)

        return predictions


    def score(self, X, y):

        predictions = self.predict(X)

        return accuracy_score(
            y,
            predictions
        )


# ===============================================================
# A2 - UNIT TESTS
# GenAI tool used: ChatGPT
# ===============================================================

class TestLab06(unittest.TestCase):


    # Test Euclidean distance
    def test_euclidean_distance(self):

        result = euclidean_distance(
            [0, 0],
            [3, 4]
        )

        self.assertEqual(
            result,
            5.0
        )


    # Test Manhattan distance
    def test_manhattan_distance(self):

        result = manhattan_distance(
            [1, 2],
            [4, 6]
        )

        self.assertEqual(
            result,
            7.0
        )


    # Test label encoding
    def test_label_encode(self):

        series = pd.Series(
            ["B", "A", "B", "C"]
        )

        encoded, mapping = label_encode(
            series
        )

        self.assertEqual(
            mapping,
            {"A": 0, "B": 1, "C": 2}
        )

        self.assertEqual(
            encoded.tolist(),
            [1, 0, 1, 2]
        )


    # Test mean imputation
    def test_mean_imputation(self):

        series = pd.Series(
            [10, np.nan, 30]
        )

        result = impute_column(
            series,
            "mean"
        )

        self.assertEqual(
            result.iloc[1],
            20
        )


    # Test median imputation
    def test_median_imputation(self):

        series = pd.Series(
            [10, np.nan, 30]
        )

        result = impute_column(
            series,
            "median"
        )

        self.assertEqual(
            result.iloc[1],
            20
        )


    # Test mode imputation
    def test_mode_imputation(self):

        series = pd.Series(
            ["A", "A", np.nan, "B"]
        )

        result = impute_column(
            series,
            "mode"
        )

        self.assertEqual(
            result.iloc[2],
            "A"
        )


    # Test bubble sort
    def test_bubble_sort(self):

        data = [5, 2, 4, 1, 3]

        result = bubble_sort(data)

        self.assertEqual(
            result,
            [1, 2, 3, 4, 5]
        )


    # Test merge sort
    def test_merge_sort(self):

        data = [5, 2, 4, 1, 3]

        result = merge_sort(data)

        self.assertEqual(
            result,
            [1, 2, 3, 4, 5]
        )


    # Test quick sort
    def test_quick_sort(self):

        data = [5, 2, 4, 1, 3]

        result = quick_sort(data)

        self.assertEqual(
            result,
            [1, 2, 3, 4, 5]
        )


    # Test majority vote
    def test_majority_vote(self):

        neighbors = [
            (1.0, "A", 0),
            (2.0, "A", 1),
            (3.0, "B", 2)
        ]

        result = majority_vote(
            neighbors
        )

        self.assertEqual(
            result,
            "A"
        )


    # Test custom kNN prediction
    def test_custom_knn(self):

        X_train = np.array([
            [0, 0],
            [0, 1],
            [5, 5],
            [5, 6]
        ])

        y_train = np.array([
            "A",
            "A",
            "B",
            "B"
        ])

        X_test = np.array([
            [1, 1],
            [5, 5]
        ])

        model = CustomKNN(k=3)

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        self.assertEqual(
            predictions[0],
            "A"
        )

        self.assertEqual(
            predictions[1],
            "B"
        )


    # Test GenAI kNN
    def test_genai_knn(self):

        X_train = np.array([
            [0, 0],
            [0, 1],
            [5, 5],
            [5, 6]
        ])

        y_train = np.array([
            "A",
            "A",
            "B",
            "B"
        ])

        X_test = np.array([
            [0, 0],
            [5, 5]
        ])

        model = GenAIKNN(k=3)

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        self.assertEqual(
            predictions[0],
            "A"
        )

        self.assertEqual(
            predictions[1],
            "B"
        )


# ===============================================================
# A3 - EVALUATION METRICS
# GenAI tool used: ChatGPT
# ===============================================================

def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    return (
        accuracy,
        precision,
        recall,
        f1
    )


# ===============================================================
# A3 - PERFORMANCE TEST
# GenAI tool used: ChatGPT
# ===============================================================

def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    runs=10
):

    times = []

    # Fit model once
    model.fit(
        X_train,
        y_train
    )

    # Predictions for metrics
    predictions = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    # Average prediction time over 10 runs
    for _ in range(runs):

        start = time.perf_counter()

        model.predict(
            X_test
        )

        end = time.perf_counter()

        times.append(
            end - start
        )

    average_time = np.mean(times)

    return (
        metrics[0],
        metrics[1],
        metrics[2],
        metrics[3],
        average_time
    )


# ===============================================================
# A3 - COMPARISON
# GenAI tool used: ChatGPT
# ===============================================================

def performance_comparison(
    X_train,
    y_train,
    X_test,
    y_test
):

    results = []


    # -----------------------------------------------------------
    # 1. Custom kNN
    # -----------------------------------------------------------

    custom_model = CustomKNN(
        k=3
    )

    result = evaluate_model(
        custom_model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    results.append([
        "Custom kNN",
        *result
    ])


    # -----------------------------------------------------------
    # 2. Scikit-learn kNN
    # -----------------------------------------------------------

    sklearn_model = KNeighborsClassifier(
        n_neighbors=3
    )

    result = evaluate_model(
        sklearn_model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    results.append([
        "Scikit-learn kNN",
        *result
    ])


    # -----------------------------------------------------------
    # 3. GenAI kNN
    # -----------------------------------------------------------

    genai_model = GenAIKNN(
        k=3
    )

    result = evaluate_model(
        genai_model,
        X_train,
        y_train,
        X_test,
        y_test
    )

    results.append([
        "GenAI kNN",
        *result
    ])


    results_df = pd.DataFrame(
        results,
        columns=[
            "Algorithm",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "Avg Time (seconds)"
        ]
    )

    return results_df


# ===============================================================
# A1 - ACCURACY VS K
# GenAI tool used: ChatGPT
# ===============================================================

def compare_accuracy_vs_k(
    X_train,
    y_train,
    X_test,
    y_test
):

    k_values = range(1, 12)

    custom_accuracy = []
    sklearn_accuracy = []
    genai_accuracy = []


    for k in k_values:

        # Custom
        custom_model = CustomKNN(
            k=k
        )

        custom_model.fit(
            X_train,
            y_train
        )

        custom_accuracy.append(
            custom_model.score(
                X_test,
                y_test
            )
        )


        # sklearn
        sklearn_model = KNeighborsClassifier(
            n_neighbors=k
        )

        sklearn_model.fit(
            X_train,
            y_train
        )

        sklearn_accuracy.append(
            sklearn_model.score(
                X_test,
                y_test
            )
        )


        # GenAI
        genai_model = GenAIKNN(
            k=k
        )

        genai_model.fit(
            X_train,
            y_train
        )

        genai_accuracy.append(
            genai_model.score(
                X_test,
                y_test
            )
        )


    plt.plot(
        list(k_values),
        custom_accuracy,
        marker="o",
        label="Custom kNN"
    )

    plt.plot(
        list(k_values),
        sklearn_accuracy,
        marker="x",
        label="Scikit-learn kNN"
    )

    plt.plot(
        list(k_values),
        genai_accuracy,
        marker="s",
        label="GenAI kNN"
    )

    plt.xlabel("k")
    plt.ylabel("Accuracy")

    plt.title(
        "Accuracy Comparison for Different k Values"
    )

    plt.legend()

    plt.show()


# ===============================================================
# MAIN
# ===============================================================

def main():

    # -----------------------------------------------------------
    # CHANGE THESE FOR YOUR PROJECT DATA
    # -----------------------------------------------------------

    df = pd.read_csv(
        "your_project_data.csv"
    )

    feature_cols = [
        "feat1",
        "feat2",
        "feat3"
    ]

    target_col = "label"


    # -----------------------------------------------------------
    # Keep two classes if dataset is multiclass
    # -----------------------------------------------------------

    classes = df[
        target_col
    ].unique()[:2]

    df = df[
        df[target_col].isin(classes)
    ]


    # -----------------------------------------------------------
    # Prepare X and y
    # -----------------------------------------------------------

    X = df[
        feature_cols
    ].to_numpy(
        dtype=float
    )

    y = df[
        target_col
    ].to_numpy()


    # -----------------------------------------------------------
    # Train-test split
    # -----------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )


    # ===========================================================
    # A1 - LAB 05 EXPERIMENTS
    # ===========================================================

    print("\n==============================")
    print("A1 - LAB 05 EXPERIMENTS")
    print("==============================")

    my_knn = CustomKNN(
        k=3
    )

    my_knn.fit(
        X_train,
        y_train
    )

    print(
        "Custom kNN accuracy:",
        my_knn.score(
            X_test,
            y_test
        )
    )

    print(
        "Predictions:",
        my_knn.predict(
            X_test
        )[:10]
    )


    sklearn_knn = KNeighborsClassifier(
        n_neighbors=3
    )

    sklearn_knn.fit(
        X_train,
        y_train
    )

    print(
        "Scikit-learn accuracy:",
        sklearn_knn.score(
            X_test,
            y_test
        )
    )


    weighted_knn = CustomKNN(
        k=3,
        weighted=True
    )

    weighted_knn.fit(
        X_train,
        y_train
    )

    print(
        "Weighted kNN accuracy:",
        weighted_knn.score(
            X_test,
            y_test
        )
    )


    # Accuracy vs k
    compare_accuracy_vs_k(
        X_train,
        y_train,
        X_test,
        y_test
    )


    # ===========================================================
    # A2 - UNIT TESTS
    # ===========================================================

    print("\n==============================")
    print("A2 - UNIT TESTS")
    print("==============================")

    test_result = unittest.TextTestRunner(
        verbosity=2
    ).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            TestLab06
        )
    )


    # ===========================================================
    # A3 - PERFORMANCE COMPARISON
    # ===========================================================

    print("\n==============================")
    print("A3 - PERFORMANCE COMPARISON")
    print("==============================")


    results = performance_comparison(
        X_train,
        y_train,
        X_test,
        y_test
    )


    print(
        results.to_string(
            index=False
        )
    )


    # Save performance table
    results.to_csv(
        "lab06_performance_results.csv",
        index=False
    )

    print(
        "\nPerformance results saved to "
        "lab06_performance_results.csv"
    )


# ===============================================================
# PROGRAM START
# ===============================================================

if __name__ == "__main__":
    main()
