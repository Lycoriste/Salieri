import os, io

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union
from soft_ac import SoftAC
import logging
import traceback

agent_info = False
inference = False

origins = [
    "http://localhost:5173", # Tracking frontend
]

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AGENT = SoftAC(state_dim=12, action_dim=2, entropy_coef=0.05, network_type="COR", batch_size=256)

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A dictionary (key: agent_name, val: agent_params)
class AgentData(BaseModel):
    agents: dict

class Observation(BaseModel):
    state: list[float]

class Experience(BaseModel):
    state: Union[list[float], list[list[float]]]
    action: Union[list[float], list[list[float]]]
    next_state: Union[list[float], list[list[float]]]
    reward: Union[float, list[float]]
    done: Union[bool, list[bool]]

def normalize_exp(exp: Experience):
    # Always return batch style (list of lists, list of floats, etc.)
    state = exp.state if isinstance(exp.state[0], list) else [exp.state]
    action = exp.action if isinstance(exp.action[0], list) else [exp.action]
    next_state = exp.next_state if isinstance(exp.next_state[0], list) else [exp.next_state]
    reward = exp.reward if isinstance(exp.reward, list) else [exp.reward]
    done = exp.done if isinstance(exp.done, list) else [exp.done]

    return state, action, next_state, reward, done

# Startup functions
@app.get("/")
async def root():
    return "Server is running"

AGENTS = []
@app.post("/init")
async def receive_init_params(request: AgentData):
    try:
        # [TODO] Implement after parse function
        for agent_name, agent_params in request.agents:
            ...
    except Exception as e:
        logger.error(f"[!] Unable to initialize agents")
        raise HTTPException(status_code=400, detail=str(e))

"""
    [POST Requests]
    Step: receive state and returns an action from policy
    Update: receive next_state after taking action and updates policy (if steps == batch_size)
"""
@app.post("/rl/step")
async def step(request: Observation):
    try:
        action = AGENT.step(request.state, deterministic=inference)
        return JSONResponse(content={"action": action})
    except Exception as e:
        logger.error(f"[!] Step error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/rl/update")
async def update(request: Experience):
    try:
        AGENT.update(normalize_exp(request))
    except Exception as e:
        logger.error(f"[!] Update error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rl/inference")
async def toggle_inference():
    try:
        inference = True
    except Exception as e:
        logger.error(f"[!] Failed to toggle inference mode")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/rl/train")
async def toggle_train():
    try:
        inference = False
    except Exception as e:
        logger.error(f"[!] Failed to toggle train mode")
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

@app.get("/graph")
async def plotly_graph():
    try:
        plots = []

        steps = AGENT.steps
        episodes = list(range(AGENT.episodes_total))
        episode_length = [AGENT.episode_length[i] for i in episodes]
        episode_reward = [AGENT.episode_reward[i] for i in episodes]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Episode length", "Episode reward"),
        )
        fig.add_trace(
            go.Scatter(x=episodes, y=episode_length, mode='lines', name='Length'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=episodes, y=episode_reward, mode='lines', name='Reward'),
            row=1, col=2
        )
        fig.update_layout(
            paper_bgcolor='#131313',
            title_font_color='white',
            font_color='white',
        )
        
        return fig.to_json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run Uvicorn to start server
if __name__ == "__main__":
    print("Server started!")
    uvicorn.run(app, host="0.0.0.0", port=7777)
