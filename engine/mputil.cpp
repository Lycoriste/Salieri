#include "net_common.h"
namespace py = pybind11;
using string = std::string;
using string_view = std::string_view;

py::dict msgpack_map_to_pydict(const msgpack::object& obj) {
  py::dict res;
  if (obj.type != msgpack::type::MAP) return res;

  for (uint32_t i = 0; i < obj.via.map.size; ++i) {
    const msgpack::object_kv& kv = obj.via.map.ptr[i];

    // Only support string keys
    if (kv.key.type != msgpack::type::STR) continue;
    string_view key(kv.key.via.str.ptr, kv.key.via.str.size);

    // Convert value depending on type
    switch (kv.val.type) {
      case msgpack::type::POSITIVE_INTEGER:
        res[py::str(key)] = static_cast<int>(kv.val.as<uint64_t>());
        break;
      case msgpack::type::NEGATIVE_INTEGER:
        res[py::str(key)] = kv.val.as<int64_t>();
        break;
      case msgpack::type::FLOAT32:
      case msgpack::type::FLOAT64:
        res[py::str(key)] = kv.val.as<double>();
        break;
      case msgpack::type::STR: {
        string_view val_str(kv.val.via.str.ptr, kv.val.via.str.size);
        res[py::str(key)] = string(val_str); // copy into Python
        break;
      }
      case msgpack::type::MAP:
        res[py::str(key)] = msgpack_map_to_pydict(kv.val); // recursive if needed
        break;
      default:
        std::cerr << "[!] Unsupported type for key: " << key << std::endl;
        break;
    }
  }

  return res;
}
