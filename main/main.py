import os
# from requests.utils import quote
# from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from main.agent import RBX_Agent
import logging
import traceback

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

## Create a new client and connect to the server
# client = MongoClient("mongodb://localhost:27017", server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
# try:
#    client.admin.command('ping')
#    print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#    print(e)

# db = client.get_database("RBX")
# collection = db['Test']

"""

    MongoDB not in use. Currently training locally.

"""

AGENT = RBX_Agent(state_space=4, action_space=6)

# FastAPI app
app = FastAPI()

# Pydantic model for request data validation
class RequestData(BaseModel):
    data: dict

class Observation(BaseModel):
    state: list[float]

class Experience(BaseModel):
    next_state: list[float]
    reward: float
    done: bool

# GET - just for fun
@app.get("/")
async def root():
    return "Server running."

"""
    [POST Requests]
    Observe: receive state and returns an action from policy
    Learn: receive next_state after taking action and updates policy
"""
@app.post("/rl/observe")
async def receive_data(request: Observation):
    try:
        action = AGENT.observe(request.state)
        return JSONResponse(content={"action": action})
    except Exception as e:
        logger.error(f"Observe Error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/rl/learn")
async def receive_data(request: Experience):
    try:
        AGENT.learn((request.next_state, request.reward, request.done))
    except Exception as e:
        logger.error(f"Learn Error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/util/save")
async def save_policy():
    try:
        AGENT.save_policy()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/util/visual")
async def visualize():
    try:
        AGENT.get_stats()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run Uvicorn to start server
if __name__ == "__main__":
    import uvicorn
    print("Server started!")
    uvicorn.run(app, host="0.0.0.0", port=0000)
