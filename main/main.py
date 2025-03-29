import os
# from requests.utils import quote
# from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from spatial_nav_agent import RBX_Agent
# Create a new client and connect to the server
client = MongoClient("mongodb://localhost:27017", server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client.get_database("RBX")
collection = db['Test']

"""

    MongoDB not in use. Currently training locally.

"""

AGENT = RBX_Agent()

# FastAPI app
app = FastAPI()

# Pydantic model for request data validation
class RequestData(BaseModel):
    data: dict

# GET
@app.get("/rl/action")
async def root():
    return "FASTAPI-BACKEND-ROOT"

# POST
@app.post("/rl/state")
async def receive_data(data: RequestData):
    try:
        AGENT.train(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run Uvicorn to start server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="Lycoris", port=7777)