# Roblox HTTP Request via Button
```lua
HTTP = game:GetService("HttpService")
SendHTTP = script.Parent:WaitForChild("SendHTTP")

POST_DATA_API_ADDRESS = "http://localhost:8000/api/data"

local data = {
	state = nil
	feature = nil
	action = nil
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