from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
from sqlalchemy.orm import Session
from pydantic import BaseModel, Json
from database import SessionLocal, engine
import models

app = FastAPI()
models.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

### Classes/tables stored and used in database ###

class RINNBase(BaseModel):
    epoch: int
    strand: int
    training_score: int
    training_output: Json

class RINNBase_data(BaseModel):
    training_data: Json

### POST requests ###

@app.post("/RINN_data/")
async def create_RINN_data(_RINN_data: RINNBase_data, db: db_dependency):
    db_RINN_data = models.RINN_data(_RINN_data.training_data)
    db.add(db_RINN_data)
    db.commit()
    db.refresh(db_RINN_data)

    return db_RINN_data