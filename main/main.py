import os, io
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from double_dqn import Double_DQN
from actor_critic import ActorCritic
import logging
import traceback

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

AGENT = ActorCritic(state_dim=5, n_step=10, entropy_coef=0.1, alpha=5e-5)
AGENT.load_policy()
AGENT.load_metadata_json()
AGENT.train()

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
    
@app.get("/util/dist")
async def plot_distribution():
    try:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(AGENT.move_p)
        plt.title("Move Probability over time")
        plt.xlabel("Step")
        plt.ylabel("P(move=1)")

        plt.subplot(1, 3, 2)
        plt.plot(AGENT.turn_mu)
        plt.title("Turn Mean (mu) over time")
        plt.xlabel("Step")
        plt.ylabel("Turn mu")

        plt.subplot(1, 3, 3)
        plt.plot(AGENT.turn_std)
        plt.title("Turn Std Dev (std) over time")
        plt.xlabel("Step")
        plt.ylabel("Turn std")

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/util/hist")
async def plot_hist():
    try:
        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        sns.histplot(AGENT.move_p[-200:], bins=50, kde=True)
        plt.title("Move action probabilities distribution")
        plt.xlabel("Probability")
        plt.ylabel("Frequency")

        plt.subplot(1,2,2)
        sns.histplot(AGENT.turn_action[-200:], bins=50, kde=True)
        plt.title("Turn action distribution")
        plt.xlabel("Probability")
        plt.ylabel("Frequency")

        plt.tight_layout()

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
