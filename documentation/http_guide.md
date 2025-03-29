# Roblox HTTP Request via Button
```lua
HTTP = game:GetService("HttpService")
SendHTTP = script.Parent:WaitForChild("SendHTTP")

POST_DATA_API_ADDRESS = "http://localhost:8000/api/data"

local data = {
	state = nil,
	reward = nil,
	done = nil,
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
Sends a HTTP request from Roblox to web server through a button click. This is as simple as it gets.

As the project scales and ideally we want to maximize training, may systems design with a load balancer to manage connections between web server and agents.

# Update feedback loop between RL agent and model
```lua
local Agent = script.Parent
local HRP = Agent:FindFirstChild("HumanoidRootPart")
local HEAD = Agent:FindFirstChild("Head")
local Humanoid = Agent:FindFirstChild("Humanoid")

local RL_Content = Agent.Parent
local Target = RL_Content:FindFirstChild("Target")

local HTTP = game:GetService("HttpService")
local SendHTTP = script.Parent:FindFirstChild("SendHTTP")	-- This is just button click for testing

-- Request action to take after reporting current state/observation
ACTION_REQ = "http://Lycoris:7777/rl/observe"
-- Report the next state after action has been taken to update policy
STATE_REQ = "http://Lycoris:7777/rl/learn"

function get_state()
	local POS = HRP.CFrame.Position
	local ROT_X, ROT_Y, ROT_Z = HRP.CFrame:ToOrientation()
	local H_ROT_X, H_ROT_Y, H_ROT_Z = HEAD.CFrame:ToOrientation()
	
	-- State representation
	local state = {
		-- HRP position (body position)
		pos_x = POS.X,
		pos_y = POS.Y,
		pos_z = POS.Z,
		-- HRP orientation (body orientation)
		rot_y = math.deg(ROT_Y),
		-- Head orientation: later update to "eyes" orientation
		--head_y = math.deg(H_ROT_Y),
	}

	return state
end

-- Append states later to complete state representation (n = 8)
local state_objects = {}

function hydrate_object()
	return
end

SendHTTP.MouseClick:Connect(function()
	print(get_state())
end)

-- Request DQN for what action to take (epsilon-greedy)
function request_action(observation)
	local result = {state = observation}
	local data = HTTP:JSONEncode(result)
	
	local success, err = pcall(function()
		HTTP:PostAsync(ACTION_REQ, data, Enum.HttpContentType.ApplicationJson, false)
	end)

	if success then
		print("Data sent successfully:", err)
	else
		warn("Failed to send data:", err)
	end
end

function report_state(state, reward, done)
	-- Format data into JSON to parse it easily later
	local result = {state, reward, done}
	local data = HTTP:JSONEncode(result)
	local success, err = pcall(function()
		HTTP:PostAsync(STATE_REQ, data, Enum.HttpContentType.ApplicationJson, false)
	end)

	if success then
		print("Data sent successfully:", err)
	else
		warn("Failed to send data:", err)
	end
end

function random_pos()
	local BoundaryLeft = RL_Content:FindFirstChild("BoundaryLeft")
	local BoundaryRight = RL_Content:FindFirstChild("BoundaryRight")

	local left_pos, right_pos = BoundaryLeft.CFrame.Position, BoundaryRight.CFrame.Position
	local rand_x = math.random(left_pos, right_pos)
	local rand_z = math.random(left_pos, right_pos)
	local current_y = HRP.CFrame.Position.Y
	
	local new_pos = CFrame.new(rand_x, current_y, rand_z)
	
	-- Implement teleportation code here
end

local done = false
Humanoid.Touched:Connect(function(touchingPart)
	if (touchingPart == Target) then
		done = true
	end
end)

while false do
	-- Default reward
	local reward = -1
	
	local state = get_state()
	-- Asks the RL model what action to take
	request_action(state)
	
	-- Observe the state it is in after action is taken
	local next_state = get_state()
	
	-- When Humanoid touches target -> done set to true (give reward)
	if (done) then
		reward = 1
	end
	
	-- Send observation once action is taken
	report_state(next_state, reward, done)
	task.wait(0.2)
end
```