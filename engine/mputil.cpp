#include "net_common.h"
#include <iostream>
#include <optional>
#include <msgpack.hpp>
#include "mputil.h"
namespace py = pybind11;
using string = std::string;
using string_view = std::string_view;



msgpack::object* get_ref(const msgpack::object& obj, const std::string_view key) {
  if (obj.type != msgpack::type::MAP) return nullptr;
  msgpack::object_kv* p = obj.via.map.ptr;
  msgpack::object_kv* const pend = obj.via.map.ptr + obj.via.map.size;
  
  for (; p < pend; ++p) {
    if (p->key.type == msgpack::type::STR) {
      std::string_view field_key(p->key.via.str.ptr, p->key.via.str.size);
      if (field_key == key) {
        try {
          return &p->val;
        } catch (std::exception& e) {
          std::cerr << "[!] Error getting field from msgpack: " << e.what() << std::endl;
        }
      }
    }
  }
  return nullptr; 
}

// Get req to keep object_handle alive
py::dict msgpack_map_to_pydict(const msgpack::object& obj) {
  py::dict res;

  if (obj.type != msgpack::type::MAP) {
    std::cerr << "[!] Params is not a map.\n";
    return res;
  }

  for (uint32_t i = 0; i < obj.via.map.size; ++i) {
    const msgpack::object_kv& kv = obj.via.map.ptr[i];

    // String keys only for explicit-ness (is that a word?)
    if (kv.key.type != msgpack::type::STR) continue;
    string_view key(kv.key.via.str.ptr, kv.key.via.str.size);
    py::str key_ = py::str(key.data(), key.size());

    // Convert value depending on type
    // Python types only to avoid segfault
    switch (kv.val.type) {
      case msgpack::type::POSITIVE_INTEGER:
      case msgpack::type::NEGATIVE_INTEGER:
        res[key_] = py::int_(kv.val.as<uint64_t>());
        break;
      case msgpack::type::FLOAT32:
      case msgpack::type::FLOAT64:
        res[key_] = py::float_(kv.val.as<double>());
        break;
      case msgpack::type::STR: {
        string_view val_view(kv.val.via.str.ptr, kv.val.via.str.size);
        res[key_] = py::str(val_view.data(), val_view.size());
        break;
      }
      case msgpack::type::MAP:
        // Recursion in case it's nested - highly unlikely and not recommended
        std::cerr << "[!] Map hit: dangerous operation.\n";
        res[key_] = msgpack_map_to_pydict(kv.val);         
        break;
      default:
        std::cerr << "[!] Unsupported type for key: " << key << std::endl;
        break;
    }
  }

  return res;
}
