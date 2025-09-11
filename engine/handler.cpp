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
py::object soft_ac_m;

void init_py() {
  py::gil_scoped_acquire acquire;
  sys_m = py::module_::import("sys");
  sys_m.attr("path").attr("insert")(0, "..");

  torch_m = py::module_::import("torch");
  soft_ac_m = py::module::import("rl.soft_ac");
  py::gil_scoped_release release;
}

// TODO: Figure out the msgpack formatting for designing NNs
void create_neural_network();

// TODO: Add authentication
void handle_start(const HttpRequest& req) {
  string_view model_name;
  try {
    auto name_opt = get_field<string_view>(req.body_view, "name");
    if (!name_opt) {
      std::cerr << "[!] Key: [name] not found or wrong type.\n";
      return;
    }

    string_view model_name = *name_opt;

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

  // Everything works -> add server to active session 
}

// Resolve save file naming conflicts
void handle_end(const HttpRequest& req) {
}

// [server_id: #, payload: {agent:{}}, inference: T/F]
void handle_step(const HttpRequest& req) {
  try {
    auto payload_opt = get_field<msgpack::object>(req.body_view, "payload");
    if (!payload_opt) {
      std::cerr << "[!] Payload not found.\n";
    }
    const msgpack::object& payload = *payload_opt;
    if (payload.type != msgpack::type::MAP) {
      std::cerr << "[!] Payload is not a map.\n";
    }

    unordered_map<string_view, std::vector<string_view>> model_to_id {};
    unordered_map<string_view, std::vector<std::vector<float>>> model_batch {};
    
    for (uint32_t i = 0; i < payload.via.map.size; ++i) {
      const msgpack::object_kv& kv = payload.via.map.ptr[i];
      if (kv.key.type != msgpack::type::STR) {
        std::cerr << "[!] Non-string key in payload.\n";
        continue;
      }
      
      // agent_id guaranteed to be unique
      string_view agent_id(kv.key.via.str.ptr, kv.key.via.str.size);
      const msgpack::object& agent_params = kv.val;

      // Potential overhead - review get_field
      auto state_opt = get_field<std::vector<float>>(agent_params, "state");
      if (!state_opt) {
        std::cerr << "[!] Missing state information for agent " << agent_id << std::endl;
        continue;
      }

      auto name_opt = get_field<string_view>(agent_params, "name");
      if (!name_opt) {
        std::cerr << "[!] Missing model name for agent " << agent_id << std::endl;
        continue;
      }
      
      model_to_id[*name_opt].push_back(agent_id);
      model_batch[*name_opt].push_back(*state_opt);
    }
    
    // Inference mode is a global setting
    bool inference_flag = false;
    auto inference_opt = get_field<bool>(req.body_view, "inference");
    if (inference_opt) {
      inference_flag = *inference_opt;
    }
    
    unordered_map<string_view, std::vector<float>> response {};

    for (const auto& [model, state_batch] : model_batch) {
      const py::list& actions = model_map[model].attr("step")(state_batch, inference_flag);
      const auto& ids = model_to_id[model];
    
      if (ids.size() != py::len(actions)){
        std::cerr << "[!] Number of agents does not match number of actions returned." << std::endl;
        return;
      }

      for (size_t i = 0; i < ids.size(); ++i) {
        response[ids[i]] = actions[i].cast<std::vector<float>>();
      }
    }
  
  } catch (const std::exception& e) {
    std::cerr << "[!] Error in handle_step: " << e.what() << std::endl;
  }
}

// Requires next_state, reward, done
void handle_update(const HttpRequest& req) {
  try {
    auto payload_opt = get_field<msgpack::object>(req.body_view, "payload");
    if (!payload_opt) {
      std::cerr << "[!] Payload not found.\n";
      return;
    }
    const msgpack::object& payload = *payload_opt;
    if (payload.type != msgpack::type::MAP) {
      std::cerr << "[!] Payload is not a map.\n";
    }

    unordered_map<string_view, std::vector<string_view>> model_to_id {};
    unordered_map<string_view, std::vector<std::vector<float>>> model_state_batch {};
    unordered_map<string_view, std::vector<std::vector<float>>> model_next_state_batch {};
    unordered_map<string_view, std::vector<float>> model_reward_batch {};
    unordered_map<string_view, std::vector<bool>> model_done_batch {};
    
    for (uint32_t i = 0; i < payload.via.map.size; ++i) {
      const msgpack::object_kv& kv = payload.via.map.ptr[i];
      if (kv.key.type != msgpack::type::STR) {
        std::cerr << "[!] Non-string key in payload.\n";
        continue;
      }

      string_view agent_id(kv.key.via.str.ptr, kv.key.via.str.size);
      const msgpack::object& agent_info = kv.val;
        
      string_view model_name {};
      std::vector<float> state {};
      std::vector<float> next_state {};
      float reward {};
      bool done {};
      
      // Check for sufficient information
      bool has_reward = false;
      bool has_done = false;
      bool has_name = false;

      for (uint32_t j = 0; j < agent_info.via.map.size; ++j) {
        const msgpack::object_kv& agent_kv = agent_info.via.map.ptr[j];
        if (agent_kv.key.type != msgpack::type::STR) {
          std::cerr << "[!] Non-string key in payload.\n";
          continue;
        }

        string_view key(agent_kv.key.via.str.ptr, agent_kv.key.via.str.size);
        const msgpack::object& val = agent_info.via.map.ptr[j].val;

        if (key == "state") {
          size_t n = val.via.array.size;
          const msgpack::object* ptr = val.via.array.ptr;
          state.resize(n);
          for (size_t k = 0; k < n; ++k) {
            state[k] = ptr[k].as<float>();
          }
        } else if (key == "next_state") {
          size_t n = val.via.array.size;
          const msgpack::object* ptr = val.via.array.ptr;
          next_state.resize(n);
          for (size_t k = 0; k < n; ++k) {
            next_state[k] = ptr[k].as<float>();
          }        
        } else if (key == "reward") {
          reward = agent_kv.val.as<float>();
          has_reward = true;
        } else if (key == "done") {
          done = agent_kv.val.as<bool>();
          has_done = true;
        } else if (key == "name") {
          model_name = agent_kv.val.as<string_view>();
          has_name = true;
        }

        if (state.empty() || next_state.empty() || !has_reward || !has_done || !has_name) {
          std::cerr << "[!] Missing one of the following arguments: state, next_state, reward, done, or name.\n";
          std::cerr << " └── Information is missing from agent: " << agent_id << std::endl;
          continue;
        }
      }

      model_to_id[model_name].push_back(agent_id);
      model_state_batch[model_name].push_back(std::move(state));
      model_next_state_batch[model_name].push_back(std::move(next_state));
      model_reward_batch[model_name].push_back(reward);
      model_done_batch[model_name].push_back(done);
    }

  } catch (std::exception& e) {
    std::cerr << "[!] Error in handle_update: " << e.what() << std::endl;
    return;
  }
}
