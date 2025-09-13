#include "net_common.h"
#include "http.h"
#include "mputil.h"
#include "debug.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

template<typename key, typename val>
using unordered_map = std::unordered_map<key, val>;
using string = std::string;
using string_view = std::string_view;
namespace py = pybind11;

// Maps user model name -> agent object
unordered_map<string, py::object> model_map {};
unordered_map<string, unordered_map<string, string>> model_metadata;
py::object sys_m;
py::object torch_m;
py::object soft_ac_m;

void init_py() {
  sys_m = py::module_::import("sys");
  sys_m.attr("path").attr("insert")(0, "..");

  torch_m = py::module_::import("torch");
  soft_ac_m = py::module::import("rl.soft_ac");
}

// TODO: Figure out the msgpack formatting for designing NNs
void create_neural_network();

// TODO: Add authentication
HttpResponse handle_start(const HttpRequest& req) {
  std::cout << "[+] Checking startup parameters...\n";
  string model_name;

  HttpResponse resp {};
  resp.status_code = HttpStatusCode::InternalServerError;
  resp.http_version = req.http_version;
  resp.content_length = 0;
  bool has_model = false;

  try {
    auto name_opt = get_field<string_view>(req.body_view, "name");
    if (!name_opt) {
      std::cerr << "[!] Key: [name] not found or wrong type.\n";
      return resp;
    }

    model_name = *name_opt;

    if (model_map.find(model_name) != model_map.end()) {
      std::cout << "[*] Model already exists: " << model_name << ". Skipping initialization.\n";
      has_model = true;
      resp.status_code = HttpStatusCode::Ok;
      return resp;
    }
  } catch (const std::exception& e) {
    std::cerr << "[!] Error reading model_name from payload: " << e.what() << std::endl;
    return resp;
  }
  
  if (!has_model) {
    // SoftAC by default
    string algorithm = "SoftAC";
    auto algorithm_opt = get_field<string>(req.body_view, "algorithm");
    if (algorithm_opt) {
      algorithm = *algorithm_opt;
    }

    auto params_opt = get_field<msgpack::object>(req.body_view, "params");
    if (!params_opt) {
      std::cerr << "[!] Missing model parameters.\n";
      return resp;
    }
    msgpack::object& params = *params_opt;
    if (params.type != msgpack::type::MAP) {
      std::cerr << "[!] Params is not a map.\n";
      return resp;
    }
    
    try {
      py::dict kwargs = msgpack_map_to_pydict(params);
      py::object agent = soft_ac_m.attr("SoftAC")(**kwargs);
      model_map[model_name] = agent;
    } catch (const std::exception& e) {
      std::cerr << "[!] Error creating model object at handle_start: " << e.what() << std::endl;
      return resp;
    }
  }

  // Everything works -> add server to active session 
  std::cout << "[+] Session created successfully." << std::endl;
  resp.status_code = HttpStatusCode::Ok;
  return resp;
}

// Resolve save file naming conflicts
HttpResponse handle_end(const HttpRequest& req) {
  HttpResponse resp {};
  resp.status_code = HttpStatusCode::InternalServerError;
  resp.http_version = req.http_version;
  resp.content_length = 0;
  
  return resp;
}

// [server_id: #, payload: {agent:{}}, inference: T/F]
HttpResponse handle_step(const HttpRequest& req) {
  HttpResponse resp {};
  resp.status_code = HttpStatusCode::InternalServerError;
  resp.http_version = req.http_version;
  resp.content_length = 0;

  try {
    auto payload_opt = get_field<msgpack::object>(req.body_view, "payload");
    if (!payload_opt) {
      std::cerr << "[!] Payload not found.\n";
      return resp;
    }
    const msgpack::object& payload = *payload_opt;
    if (payload.type != msgpack::type::MAP) {
      std::cerr << "[!] Payload is not a map.\n";
      return resp;
    }

    unordered_map<string, std::vector<string>> model_to_id {};
    unordered_map<string, std::vector<std::vector<float>>> model_batch {};
    
    for (uint32_t i = 0; i < payload.via.map.size; ++i) {
      const msgpack::object_kv& kv = payload.via.map.ptr[i];
      if (kv.key.type != msgpack::type::STR) {
        std::cerr << "[!] Non-string key in payload.\n";
        continue;
      }
      
      // agent_id guaranteed to be unique
      string agent_id(kv.key.via.str.ptr, kv.key.via.str.size);
      const msgpack::object& agent_params = kv.val;

      // Potential overhead - review get_field
      auto state_opt = get_field<std::vector<float>>(agent_params, "state");
      if (!state_opt) {
        std::cerr << "[!] Missing state information for agent " << agent_id << std::endl;
        continue;
      }

      auto name_opt = get_field<string>(agent_params, "name");
      if (!name_opt) {
        std::cerr << "[!] Missing model name for agent " << agent_id << std::endl;
        continue;
      }
      
      model_to_id[*name_opt].push_back(agent_id);
      model_batch[*name_opt].push_back(*state_opt);
    }

    pass("Batched states to respective models.");
    
    // Inference mode is a global setting
    bool inference_flag = false;
    auto inference_opt = get_field<bool>(req.body_view, "inference");
    if (inference_opt) {
      inference_flag = *inference_opt;
    }
    
    unordered_map<string, std::vector<float>> response {};

    for (const auto& [model, state_batch] : model_batch) {
      print_mat(state_batch);
      py::array_t<float> mat = vec_to_numpy(state_batch);
      pass("vec_to_numpy ran successfully.");

      auto model_it = model_map.find(model);
      if (model_it == model_map.end()) {
        err("Model " + string(model) + " is not found in map.");
        return resp;
      }
      pass("Model " + string(model) + " is found in map.");

      if (model_it->second.is_none()) {
        err("Model " + string(model) + " is no longer a valid Python object.");
        return resp;
      }
      pass("Model " + string(model) + " exists in memory.");

      const py::list& actions = model_map[model].attr("step")(mat, inference_flag);
      const auto& ids = model_to_id[model];
    
      if (ids.size() != py::len(actions)){
        err("Number of agents does not match number of actions returned.");
        return resp;
      }

      pass("Action inference passed.");
      std::vector<std::vector<float>> actions_vec = actions.cast<std::vector<std::vector<float>>>();

      for (size_t i = 0; i < ids.size(); ++i) {
        response[ids[i]] = actions_vec[i];
      }
    }

    msgpack::sbuffer sbuf;
    msgpack::pack(sbuf, response);
    
    resp.content = std::string(sbuf.data(), sbuf.size());
    resp.content_length = sbuf.size();
    resp.status_code = HttpStatusCode::Ok;

    return resp;

  } catch (const std::exception& e) {
    std::cerr << "\033[31m[!] Error in handle_step: " << e.what() << "\033[0m" << std::endl;
    return resp;
  }
}

// Requires next_state, reward, done
HttpResponse handle_update(const HttpRequest& req) {
  HttpResponse resp {};
  resp.http_version = req.http_version;
  resp.status_code = HttpStatusCode::InternalServerError;
  resp.content_length = 0;

  try {
    auto payload_opt = get_field<msgpack::object>(req.body_view, "payload");
    if (!payload_opt) {
      err("Payload not found.");
      return resp;
    }
    const msgpack::object& payload = *payload_opt;
    if (payload.type != msgpack::type::MAP) {
      err("Payload is not a map.");
      return resp;
    }

    unordered_map<string, std::vector<string_view>> model_to_id {};
    unordered_map<string, std::vector<std::vector<float>>> model_state_batch {};
    unordered_map<string, std::vector<std::vector<float>>> model_action_batch {};
    unordered_map<string, std::vector<std::vector<float>>> model_next_state_batch {};
    unordered_map<string, std::vector<float>> model_reward_batch {};
    unordered_map<string, std::vector<bool>> model_done_batch {};
    
    for (uint32_t i = 0; i < payload.via.map.size; ++i) {
      const msgpack::object_kv& kv = payload.via.map.ptr[i];
      if (kv.key.type != msgpack::type::STR) {
        err("Non-string key in payload.");
        continue;
      }

      string_view agent_id(kv.key.via.str.ptr, kv.key.via.str.size);
      const msgpack::object& agent_info = kv.val;
        
      string model_name {};
      std::vector<float> state {};
      std::vector<float> action {};
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
          err("Non-string key in payload.");
          continue;
        }

        string_view key(agent_kv.key.via.str.ptr, agent_kv.key.via.str.size);
        const msgpack::object& val = agent_info.via.map.ptr[j].val;
        
        // I'm sorry for this monstrosity
        if (key == "state") {
          if (val.type == msgpack::type::ARRAY) {
            state = val.as<std::vector<float>>();
            pass("State passed.");
          } else {
            state = { val.as<float>() };
          }

        } else if (key == "action") {
          if (val.type == msgpack::type::ARRAY) {
            action = val.as<std::vector<float>>();
            pass("Action passed.");
          } else {
            action = { val.as<float>() };
          }

        } else if (key == "next_state") {
          if (val.type == msgpack::type::ARRAY) {
            next_state = val.as<std::vector<float>>();
            pass("Next_state passed.");
          } else {
            next_state = { val.as<float>() };
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
      }

      if (state.empty() || action.empty() || next_state.empty() || !has_reward || !has_done || !has_name) {
          err("Missing one of the following arguments: state, next_state, reward, done, or name.");
          if (state.empty())
            std::cerr << " └── state is missing from agent: " << agent_id << std::endl;
          if (action.empty())
            std::cerr << " └── action is missing from agent: " << agent_id << std::endl;
          if (next_state.empty())
            std::cerr << " └── next_state is missing from agent: " << agent_id << std::endl;
          if (!has_reward)
            std::cerr << " └── reward is missing from agent: " << agent_id << std::endl;
          if (!has_done)
            std::cerr << " └── done is missing from agent: " << agent_id << std::endl;
          if (!has_name)
            std::cerr << " └── name is missing from agent: " << agent_id << std::endl;
          continue;
      }

      model_to_id[model_name].push_back(agent_id);
      model_state_batch[model_name].push_back(std::move(state));
      model_action_batch[model_name].push_back(std::move(action));
      model_next_state_batch[model_name].push_back(std::move(next_state));
      model_reward_batch[model_name].push_back(reward);
      model_done_batch[model_name].push_back(done);
    }
    
    // Nothing to send back - just update the model
    for (const auto& [model_name, _] : model_to_id) {
        const auto& state = model_state_batch[model_name];
        const auto& action = model_action_batch[model_name];
        const auto& next_state = model_next_state_batch[model_name];
        const auto& reward = model_reward_batch[model_name];
        const auto& done = model_done_batch[model_name];

        py::tuple data = py::make_tuple(state, action, next_state, reward, done);
        model_map[model_name].attr("update")(data);
    }
    
    resp.status_code = HttpStatusCode::Ok;
    return resp;

  } catch (std::exception& e) {
    std::cerr << "[!] Error in handle_update: " << e.what() << std::endl;
    return resp;
  }
}
