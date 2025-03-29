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

# GET - just for fun
@app.get("/")
async def root():
    return "FASTAPI-BACKEND-ROOT"

# POST requests
"""

    Observe: receive state and returns an action from policy
    Learn: receive next_state after taking action and updates policy

"""
@app.post("/rl/observe")
async def receive_data(data: RequestData):
    try:
        AGENT.observe(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/rl/learn")
async def receive_data(data: RequestData):
    try:
        AGENT.learn(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rl/save")
async def save_policy():
    try:
        AGENT.save_policy()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rl/visualize")
async def visualize():
    try:
        AGENT.get_stats()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run Uvicorn to start server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="Lycoris", port=7777)