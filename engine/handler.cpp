#include "net_common.h"
#include "http.h"
#include "handler.h"

namespace py = pybind11;

// Maps user model name -> agent object
std::unordered_map<std::string, py::object> model_map {};
py::object torch_module;
py::object soft_ac_module;

void init_py() {
  torch_module = py::module_::import("torch");
  soft_ac_module = py::module::import("rl.soft_ac");
}

// TODO: Figure out the msgpack formatting for designing NNs
void create_neural_network() {
}

// Load save (Y/n)
// Model name
void handle_start(const HttpRequest& req) {
  const std::string& model_name = req.body.at("name").as<std::string>();
  if (model_map.find(model_name) != model_map.end()) {
    std::cerr << "[*] Model already exists: " << model_name << ". Skipping initialization.\n";
    return;
  }
  
  std::string algorithm = "SoftAC";
  if (req.body.contains("algorithm")) {
    algorithm = req.body.at("algorithm").as<std::string>();
  }
  const std::unordered_map<std::string, int>& model_params = req.body.at("params").as<std::unordered_map<std::string, int>>();

  try {
    py::dict kwargs;
    for (auto& [param, value] : model_params) {
      kwargs[py::str(param)] = value;
    }

    py::object agent = soft_ac_module.attr("SoftAC")(**kwargs);
    model_map[model_name] = agent;
  } catch (const std::exception& e) {
    std::cerr << "[!] Error creating agent object at handle_start: " << e.what() << std::endl;
  }
}

void handle_end(const HttpRequest& req) {
}


// [server_id: #, payload: {agent:{}}, inference: T/F]
void handle_step(const HttpRequest& req) {
  try {
    auto& payload_obj = req.body.at("payload");
    const auto& payload = payload_obj.as<std::unordered_map<std::string, msgpack::object>>();
    std::vector<std::string> agent_ids;
    std::vector<std::vector<float>> state_batch {};
    agent_ids.reserve(payload.size());
    state_batch.reserve(payload.size());

    for (auto& [agent_id, agent_obj] : payload) {
      auto agent_info = agent_obj.as<std::unordered_map<std::string, msgpack::object>>();
      auto state = agent_info["state"].as<std::vector<float>>();
      agent_ids.push_back(agent_id);
      state_batch.push_back(std::move(state));
    }

    bool inference_flag = false;
    if (req.body.contains("inference")) {
      inference_flag = req.body.at("inference").as<bool>();
    }

    soft_ac_module.attr("step")(state_batch, inference_flag);
  } catch (const std::exception& e) {
    std::cerr << "[!] Error in handle_step: " << e.what() << std::endl;
  }
}

void handle_update(const HttpRequest& req) {
}
