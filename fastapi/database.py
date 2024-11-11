from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

URL_DATABASE = 'postgresql://lycoris:rlgdaMwA9P7kPiz5jw8N@frontierdb.cb6cycsu2kgm.us-west-1.rds.amazonaws.com:5432/modeldb'

engine = create_engine(URL_DATABASE)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()