from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from sqlalchemy.orm import Session
from pydantic import BaseModel, Json
from database import SessionLocal, engine
import models

app = FastAPI()

class RINNBase(BaseModel):
    epoch: int
    strand: int
    training_score: int
    training_output: Json

class RINNBase_Data(BaseModel):
    training_data: Json