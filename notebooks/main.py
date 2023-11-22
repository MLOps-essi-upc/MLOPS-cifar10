from typing import Union

from fastapi import FastAPI

app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/test/{item_id}")
def read_root(item_id: int):
    x = item_id*2
    return {"Hetestllo": x}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}