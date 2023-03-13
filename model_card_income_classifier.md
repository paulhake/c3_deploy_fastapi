# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf
This model is to predict the income category of an individual based on census data

## Model Details
A random forest classifier based on sci-kit learn library, trained on census data and predicts whether an individual has a high salary (> $50K) or a low salary (< $50K>)
Model trained using one hot encoding for categorical variables, an 80/20 train test split. Model is also provided with an API using FastApi framework and deployed to Heroku for online realtime inferencing

## Intended Use
Model used to predict likely income category based on census data
## Training Data
32560*0.8 records with the following features: age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, salary.
Training data sourced from: [UCI Census Data](https://archive.ics.uci.edu/ml/datasets/census+income)

## Evaluation Data
32560*0.2 records with the following features: age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, salary
## Metrics
Metrics used and your model's performance on those metrics.
Precision:  0.71. Recall:  0.61. Fbeta:  0.66

## Ethical Considerations
Model should be evaluated for fairness or bias given it has gender, race and age as input features

## Caveats and Recommendations
Model tested on a narrow slice of historical census data and may not generalize.