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