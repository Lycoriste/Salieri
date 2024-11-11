from database import Base
from sqlalchemy import Column, Integer, String, JSON, Boolean, Float
from database import Base

### Reinforcement Inferencing Neural Network (RINN) training results
class RINN(Base):
    __tablename__ = 'training_results'

    id = Column(Integer, primary_key=True, index=True)
    epoch = Column(Integer, primary_key=True, index=True)
    strand = Column(Integer, index=True)
    training_score = Column(Integer)
    training_output = Column(JSON)

### Table for storing data retrieved from client
class RINN_data(Base):
    __tablename__ = 'training_data'

    id = Column(Integer, primary_key=True, index=True)
    training_data = Column(JSON)