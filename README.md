# RBXWebserver - Bridging Machine Learning and Roblox Studio

This web server allows Roblox Studio to access external APIs and libraries. Made for reinforcement learning research inside a virtual environment.

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

# Roblox HTTP Handler DEMO via Button
```lua
HTTP = game:GetService("HttpService")
SendHTTP = script.Parent:WaitForChild("SendHTTP")

POST_DATA_API_ADDRESS = "http://localhost:8000/api/RINN_data"
SEND_DATA = "RINN_data"

local data = {
	training_data = {
		feature1 = 1,
		feature2 = 2
	}
}

local jsonData = HTTP:JSONEncode(data)

SendHTTP.MouseClick:Connect(function(player) 
	print("Click connected:")
	print(jsonData)
	
	local success, err = pcall(function()
		HTTP:PostAsync(POST_DATA_API_ADDRESS, jsonData, Enum.HttpContentType.ApplicationJson, false)
	end)
	
	if success then
		print("Data sent successfully:", err)
	else
		warn("Failed to send data:", err)
	end
end)
```
Sends a HTTP request from Roblox to web server through a button click.
