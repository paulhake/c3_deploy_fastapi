# Put the code for your API here.
import os
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from online_inference import online_inference

app = FastAPI()

#list of categorical features
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

#pydantic data model for online input
class RowData(BaseModel):
    age: int = Field(..., example=32)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=205019)
    education: str = Field(..., example="Assoc-acdm")
    education_num: int = Field(..., example=12)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Sales")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=50)
    native_country: str = Field(..., example="United-States")

#config for DVC and Heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system('rm -rf .dvc/cache')
    os.system('rm -rf .dvc/tmp/lock')
    os.system('dvc config core.hardlink_lock true')
    if os.system("dvc pull -q") != 0:
        exit("dvc pull failed")
    os.system("rm -rf .dvc .apt/usr/lib/dvc")

#API functions
@app.get("/")
def home():
    return {"Hello": "Welcome to project 3 - Model deployment and testing"}

#inference api
@app.post('/inference')
async def predict_income(inputrow: RowData):
    row_dict = jsonable_encoder(inputrow)
    preds = online_inference(row_dict=row_dict,cat_features=cat_features)
    
    return {"income class": preds}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)