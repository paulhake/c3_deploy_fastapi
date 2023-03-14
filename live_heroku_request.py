import requests

row = {
    "age": 47,
    "workclass": "Private",
    "fnlgt": 51835,
    "education": "Prof-school",
    "education_num": 15,
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Wife",
    "race": "White",
    "sex": "Female",
    "capital_gain": 0,
    "capital_loss": 1902,
    "hours_per_week": 60,
    "native_country": "Honduras"
}
applink = 'https://demo-app3.herokuapp.com/inference'
print(type(row))
response = requests.post(
    url=applink,
    json=row
)
assert response.status_code==200
print(response.status_code)
print(response.json())
