# Put the code for your API here.
from fastapi import FastAPI
from typing import Union, List
from pydantic import BaseModel

#instantiate the app
app = FastAPI()

class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]
    item_id: int

#define a GET
@app.get("/")
async def say_hello():
    return {"greeting":"Hello now World"}

@app.post("/items/")
async def creat_item(item: TaggedItem):
    return item

@app.get("/items/{item_id}")
async def get_items(item_id: int, count: int=1):
    return {"fetch": f"Fetched {count} of {item_id}"}

class Value(BaseModel):
    value: int

# Use POST action to send data to the server
@app.post("/{path}")
async def exercise_function(path: int, query: int, body: Value):
    return {"path": path, "query": query, "body": body}