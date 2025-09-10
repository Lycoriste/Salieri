#include "net_common.h"
#include "http.h"
#include "mputil.h"

template<typename key, typename val>
using unordered_map = std::unordered_map<key, val>;
using string = std::string;
using string_view = std::string_view;
namespace py = pybind11;

// Maps user model name -> agent object
unordered_map<string_view, py::object> model_map {};
unordered_map<string_view, unordered_map<string, string>> model_metadata;
py::object sys_m;
py::object torch_m;
py::object replay_memory_m;
py::object soft_ac_m;

void init_py() {
  sys_m = py::module_::import("sys");
  sys_m.attr("path").attr("insert")(0, "..");

  torch_m = py::module_::import("torch");
  replay_memory_m = py::module_::import("rl.replay_memory");
  soft_ac_m = py::module::import("rl.soft_ac");
}

// TODO: Figure out the msgpack formatting for designing NNs
void create_neural_network();

// TODO: Add authentication
void handle_start(const HttpRequest& req) {
  std::cout << "Handle start:\n";
  string_view model_name;
  try {
    auto model_name_opt = get_field<string_view>(req.body_view, "model_name");
    if (!model_name_opt) {
      std::cerr << "[!] model_name key not found or wrong type.\n";
      return;
    }

    string_view model_name = *model_name_opt;

    if (model_map.find(model_name) != model_map.end()) {
      std::cout << "[*] Model already exists: " << model_name << ". Skipping initialization.\n";
      return;
    } else {
      std::cout << "[+] Created model: " << model_name << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "[!] Error reading model_name from payload: " << e.what() << std::endl;
    return;
  }
  
  // SoftAC by default
  string algorithm = "SoftAC";
  auto algorithm_opt = get_field<string>(req.body_view, "algorithm");
  if (algorithm_opt) {
    algorithm = *algorithm_opt;
  }

  auto params_opt = get_field<msgpack::object>(req.body_view, "params");
  if (!params_opt) {
    std::cerr << "[!] Missing model parameters.\n";
  }
  msgpack::object params = *params_opt;
  if (params.type != msgpack::type::MAP) {
    std::cerr << "[!] Params is not a map.\n";
    return;
  }

  try {
    py::dict kwargs = msgpack_map_to_pydict(params);
    py::object agent = soft_ac_m.attr("SoftAC")(**kwargs);
    model_map[model_name] = agent;
  } catch (const std::exception& e) {
    std::cerr << "[!] Error creating model object at handle_start: " << e.what() << std::endl;
  }
}

// Resolve save file naming conflicts
void handle_end(const HttpRequest& req) {
}

// [server_id: #, payload: {agent:{}}, inference: T/F]
void handle_step(const HttpRequest& req) {
  std::cout << "what";
  return;
}
//void handle_step(const HttpRequest& req) {
//  try {
//    auto& payload_obj = req.body.at("payload");
//    const auto& payload = payload_obj.as<unordered_map<string, msgpack::object>>();
//    std::vector<string> agent_ids;
//    std::vector<std::vector<float>> state_batch {};
//    agent_ids.reserve(payload.size());
//    state_batch.reserve(payload.size());
//
//    for (auto& [agent_id, agent_obj] : payload) {
//      auto agent_info = agent_obj.as<unordered_map<string, msgpack::object>>();
//      auto state = agent_info["state"].as<std::vector<float>>();
//      agent_ids.push_back(agent_id);
//      state_batch.push_back(std::move(state));
//    }
//
//    bool inference_flag = false;
//    if (req.body.contains("inference")) {
//      inference_flag = req.body.at("inference").as<bool>();
//    }
//
//    soft_ac_m.attr("step")(state_batch, inference_flag);
//  } catch (const std::exception& e) {
//    std::cerr << "[!] Error in handle_step: " << e.what() << std::endl;
//  }
//}

void handle_update(const HttpRequest& req) {
}
