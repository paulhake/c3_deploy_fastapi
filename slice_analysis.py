"""
This module calculates model performance on slices of the data for categorical features.
"""
import os
import joblib
import pandas as pd
from data import process_data
from model import compute_model_metrics

data_path = 'data/cleaned_census.csv'
artifacts_path = 'model'

# Categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
# read model artifacts
model = joblib.load(os.path.join(artifacts_path, 'model.pkl'))
encoder = joblib.load(os.path.join(artifacts_path, 'encoder.pkl'))
lb = joblib.load(os.path.join(artifacts_path, 'lb.pkl'))


def test_performance():
    # read data and model artifacts
    data = pd.read_csv(data_path)

    slice_metrics = []
    # slice data by categorical feature
    for feature in cat_features:
        for cls in data[feature].unique():
            df_test = data[data[feature] == cls]

            X_test, y_test, _, _ = process_data(
                df_test,
                cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                training=False)

            preds = model.predict(X_test)
            # calc metrics for slice
            precision, recall, fbeta = compute_model_metrics(y_test, preds)
            row = f"{feature} - {cls} :: Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}"
            slice_metrics.append(row)

            with open('slice_output.txt', 'w') as file:
                for row in slice_metrics:
                    file.write(row + '\n')


if __name__ == '__main__':
    test_performance()
