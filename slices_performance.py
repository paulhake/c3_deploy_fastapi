"""
This module outputs the performance of the model on slices of the data for categorical features.
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
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

def test_performance():
    """ Check performance by categorical features """

    data = pd.read_csv(data_path)
    _, test = train_test_split(data, test_size=0.20)

    rf = joblib.load(os.path.join(artifacts_path, 'model.pkl'))
    encoder = joblib.load(os.path.join(artifacts_path, 'encoder.pkl'))
    lb = joblib.load(os.path.join(artifacts_path, 'lb.pkl'))

    slice_metrics = []

    for feature in cat_features:
        for cls in test[feature].unique():
            df_temp = test[test[feature] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp,
                cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                training=False)

            y_pred = rf.predict(X_test)

            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
            row = f"{feature} - {cls} :: Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}"
            slice_metrics.append(row)

            with open('slice_output.txt', 'w') as file:
                for row in slice_metrics:
                    file.write(row + '\n')



if __name__ == '__main__':
    test_performance()