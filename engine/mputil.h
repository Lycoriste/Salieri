#pragma once
#include <iostream>
#include <optional>
#include <msgpack.hpp>
#include <pybind11/embed.h>

// msgpack zero-copy view
template<typename T>
std::optional<T> get_field(const msgpack::object& obj, const std::string_view key) {
  if (obj.type != msgpack::type::MAP) return std::nullopt;
  msgpack::object_kv* p = obj.via.map.ptr;
  msgpack::object_kv* const pend = obj.via.map.ptr + obj.via.map.size;
  
  for (; p < pend; ++p) {
    if (p->key.type == msgpack::type::STR) {
      std::string_view field_key(p->key.via.str.ptr, p->key.via.str.size);
      if (field_key == key) {
        try {
          return p->val.as<T>();
        } catch (std::exception& e) {
          std::cerr << "[!] Error getting field from msgpack: " << e.what() << std::endl;
          return std::nullopt;
        }
      }
    }
  }
  return std::nullopt;
}


inline pybind11::dict msgpack_map_to_pydict(const msgpack::object& obj) {
  pybind11::dict res;

  if (obj.type != msgpack::type::MAP) {
    std::cerr << "[!] Params is not a map.\n";
    return res;
  }

  for (uint32_t i = 0; i < obj.via.map.size; ++i) {
    const msgpack::object_kv& kv = obj.via.map.ptr[i];

    // String keys only for explicit-ness (is that a word?)
    if (kv.key.type != msgpack::type::STR) continue;
    std::string_view key(kv.key.via.str.ptr, kv.key.via.str.size);
    pybind11::str key_ = pybind11::str(key.data(), key.size());

    // Convert value depending on type
    // Python types only to avoid segfault
    switch (kv.val.type) {
      case msgpack::type::POSITIVE_INTEGER:
      case msgpack::type::NEGATIVE_INTEGER:
        res[key_] = pybind11::int_(kv.val.as<uint64_t>());
        break;
      case msgpack::type::FLOAT32:
      case msgpack::type::FLOAT64:
        res[key_] = pybind11::float_(kv.val.as<double>());
        break;
      case msgpack::type::STR: {
        std::string_view val_view(kv.val.via.str.ptr, kv.val.via.str.size);
        res[key_] = pybind11::str(val_view.data(), val_view.size());
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


inline std::string b64_encode(const std::string& data) {
  static const char b64_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const unsigned char* bytes = reinterpret_cast<const unsigned char*>(data.data());
  size_t len = data.size();

  std::string encoded;
  encoded.reserve(((len + 2) / 3) * 4);

  for (size_t i = 0; i < len;) {
    uint32_t octet_a = i < len ? bytes[i++] : 0;
    uint32_t octet_b = i < len ? bytes[i++] : 0;
    uint32_t octet_c = i < len ? bytes[i++] : 0;
    uint32_t triple = (octet_a << 16) | (octet_b << 8) | octet_c;

    encoded.push_back(b64_table[(triple >> 18) & 0x3F]);
    encoded.push_back(b64_table[(triple >> 12) & 0x3F]);
    encoded.push_back(i > len + 1 ? '=' : b64_table[(triple >> 6) & 0x3F]);
    encoded.push_back(i > len ? '=' : b64_table[triple & 0x3F]);
  }

  return encoded;
}
