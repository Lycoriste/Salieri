local Agent = script.Parent
local HRP = Agent:FindFirstChild("HumanoidRootPart")
local HEAD = Agent:FindFirstChild("Head")
local Humanoid = Agent:FindFirstChild("Humanoid")

local RL_Content = Agent.Parent
local Target = RL_Content:FindFirstChild("Target")

local HTTP = game:GetService("HttpService")
local SendHTTP = script.Parent:FindFirstChild("SendHTTP")	-- This is just button click for testing

local hostname = "localhost"
local port = 7777
-- Request action to take after reporting current state/observation
local ACTION_REQ = "http://" .. hostname .. ":" .. port .. "/rl/observe"
-- Report the next state after action has been taken to update policy
local STATE_REQ = "http://" .. hostname .. ":" .. port .. "/rl/learn"

local wallFilter = RaycastParams.new()
wallFilter.FilterType = Enum.RaycastFilterType.Exclude
wallFilter.FilterDescendantsInstances = {game.Workspace.RL_Content.Room.Part}

local function get_wall_distance()
	local origin = HRP.Position
	local direction = HRP.CFrame.LookVector * 50

	local result = workspace:Raycast(origin, direction, wallFilter)
	if result then
		return (result.Position - origin).Magnitude
	else
		return 50
	end
end

local function distance_from_target()
	local origin = HRP.Position
	local distance = (HRP.Position - Target.Position).Magnitude
	return distance
end

local function get_alignment()
	local POS = HRP.CFrame.Position
	local TARGET = Target.Position
	
	local agent_look = HRP.CFrame.LookVector
	local target_direction = (TARGET - POS).Unit
	local alignment = agent_look:Dot(target_direction)
	
	return alignment
end

function get_state()
	local POS = HRP.CFrame.Position
	local TARGET = Target.Position
	local ROT_X, ROT_Y, ROT_Z = HRP.CFrame:ToOrientation()
	local H_ROT_X, H_ROT_Y, H_ROT_Z = HEAD.CFrame:ToOrientation()
	local TARGET_DIST = distance_from_target()
	
	local yaw = ROT_Y
	local yaw_sin = math.sin(yaw)
	local yaw_cos = math.cos(yaw)
	
	local REL_X = POS.X - TARGET.X
	local REL_Y = POS.Y - TARGET.Y
	local REL_Z = POS.Z - TARGET.Z
	
	local agent_forward = HRP.CFrame.LookVector
	
	local target_direction = (TARGET - POS).Unit
	local dot_product = agent_forward:Dot(target_direction)
	local cross_product = agent_forward:Cross(target_direction)
	
	local relative_yaw_sin = cross_product.Y  -- + = target is to the right
	local relative_yaw_cos = dot_product 
	
	local agent_right = HRP.CFrame.RightVector
	local target_local_x = target_direction:Dot(agent_right)     -- How much target is to our right
	local target_local_z = target_direction:Dot(agent_forward)   -- How much target is in front
	
	
	-- State representation
	local state = {
		REL_X,
		REL_Y,
		REL_Z,
		target_local_x,
		target_local_z,
	}

	return state
end

SendHTTP.MouseClick:Connect(function()
	print(get_state())
end)

-- Request DQN for what action to take (epsilon-greedy)
function request_action(observation)
	local result = {
		state = observation
	}
	local data = HTTP:JSONEncode(result)
	
	local success, res = pcall(function()
		return HTTP:PostAsync(ACTION_REQ, data, Enum.HttpContentType.ApplicationJson, false)
	end)

	if success then
		return res
	else
		warn("Failed to send data:", res)
		return nil
	end
end

function report_state(state, reward, done)
	-- Format data into JSON to parse it easily later
	local experience = {
		next_state = state, 
		reward = reward, 
		done = done
	}
	local data = HTTP:JSONEncode(experience)
	
	local _, err = pcall(function()
		HTTP:PostAsync(STATE_REQ, data, Enum.HttpContentType.ApplicationJson, false)
	end)

	if err then
		warn("Failed to send data:", err)
	end
end

function random_pos()
	local BoundaryLeft = RL_Content:FindFirstChild("BoundaryLeft")
	local BoundaryRight = RL_Content:FindFirstChild("BoundaryRight")

	local left_pos, right_pos = BoundaryLeft.CFrame.Position, BoundaryRight.CFrame.Position
	local min_x = math.min(left_pos.X, right_pos.X)
	local max_x = math.max(left_pos.X, right_pos.X)
	local min_z = math.min(left_pos.Z, right_pos.Z)
	local max_z = math.max(left_pos.Z, right_pos.Z)
	
	local rand_x = math.random(min_x, max_x)
	local rand_z = math.random(min_z, max_z)
	local current_y = Target.Position.Y
	
	local new_pos = Vector3.new(rand_x, current_y, rand_z)
	
	-- Implement teleportation code here
	Target.Position = new_pos
end

local done = false
local reward = -1
Humanoid.Touched:Connect(function(part)
	if (part == Target) then
		print("Target found")
		if not done then
			done = true
		else
			return
		end
		
		Humanoid:Move(Vector3.new(0, 0, 0), false)
		reward = 1000
		
		local next_state = get_state()
		report_state(next_state, reward, done)
		print(reward)
		
		next_state = nil
		done = false
		random_pos()
	elseif part.Parent == game.Workspace.RL_Content:FindFirstChild("Room") then
		reward -= 100
	end
end)

while true do	
	local state = get_state()
	local json_action = request_action(state)
	local time_penalty = -1
	local distance_reward = 0
	local direction_reward = 0
	
	local steering_val
	local moving_val
	
	local d = distance_from_target()
	
	if json_action then
		local action = HTTP:JSONDecode(json_action)['action']
		moving_val = action[1]
		steering_val = action[2]

		if (steering_val ~= 0) then
			Humanoid.AutoRotate = false
			HRP.CFrame = HRP.CFrame * CFrame.Angles(0, steering_val, 0)
		else
			Humanoid.AutoRotate = true
		end
		
		if moving_val == 1 then
			Humanoid:Move(HRP.CFrame.lookVector, false)  -- Move forward in world space
		end
	end
	
	task.wait(0.25)
	Humanoid:Move(Vector3.new(0,0,0), false)
	
	if not done then
		local next_state = get_state()
		local alignment = get_alignment()
		--local next_alignment = next_state[#next_state - 1]
		
		distance_reward = (d - distance_from_target()) * 3
		direction_reward = alignment * 2
		
		reward = time_penalty + distance_reward + direction_reward
		
		print(reward)
		report_state(next_state, reward, done)
	end

	next_state = nil
end