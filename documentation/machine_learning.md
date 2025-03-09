# Base neural network prototype
```python
class RINN(Base):
    __tablename__ = 'training_results'

    id = Column(Integer, primary_key=True, index=True)
    epoch = Column(Integer, primary_key=True, index=True)
    strand = Column(Integer, index=True)
    training_score = Column(Integer)
    training_output = Column(JSON)

    def __init(self, epoch, strand, training_score, training_output):
        self.epoch = epoch
        self.strand = strand
        self.training_score = training_score
        self.training_output = training_output

class RINN_data(Base):
    __tablename__ = 'training_data'

    id = Column(Integer, primary_key=True, index=True)
    training_data = Column(JSON)

    def __init__(self, training_data):
        self.training_data = training_data
```
Takes in JSON data from roblox to reduce the amount of columns required to store data in database. Reduces complexity in database management.
