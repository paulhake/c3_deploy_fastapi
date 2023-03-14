import json

import requests

r = requests.get('https://demo-app3.herokuapp.com/')

print(r.status_code)
print(r.json())

data = {
    "age": 33,
    "workclass": "Private",
    "fnlgt": 22222,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married_civ_spouse",
    "occupation": "Prof_specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

# row = {}
# for k, v in data.items():
#     row[k.replace("-", '_')] = v
#
# print(row)

r = requests.post('https://demo-app3.herokuapp.com/inference', json=data)
#
print(r.status_code)
print(r.json())