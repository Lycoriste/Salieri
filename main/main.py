import os, io
import matplotlib.pyplot as plt
# from requests.utils import quote
# from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from agent import RBX_Agent
import logging
import traceback

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AGENT = RBX_Agent(state_space=6, epsilon_start=0.9, epsilon_decay=100, epsilon_end=0.01)
AGENT.load_policy(episode_num=61)

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

@app.get("/util/save")
async def save_policy_get():
    try:
        AGENT.save_policy()
        return {"detail": "Policy saved via GET (for testing)"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/util/stats")
async def stats():
    try:
        return AGENT.get_stats()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/util/plot_reward")
async def plot_reward():
    try:
        plt.figure(figsize=(10,5))
        episodes = list(range(AGENT.episodes_total))
        rewards = [AGENT.episode_reward[i] for i in episodes]
        plt.plot(episodes, rewards, label="Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/util/plot_length")
async def plot_length():
    try:
        plt.figure(figsize=(10,5))
        episodes = list(range(AGENT.episodes_total))
        length = [AGENT.episode_length[i] for i in episodes]
        plt.plot(episodes, length, label="Episode Length")
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run Uvicorn to start server
if __name__ == "__main__":
    import uvicorn
    print("Server started!")
    uvicorn.run(app, host="0.0.0.0", port=7777)
