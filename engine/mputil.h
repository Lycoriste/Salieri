#pragma once
#include <iostream>
#include <optional>
#include <msgpack.hpp>
#include <pybind11/embed.h>

// msgpack zero-copy view
template<typename T>
std::optional<T> get_field(const msgpack::object& obj, const std::string_view& key) {
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

pybind11::dict msgpack_map_to_pydict(const msgpack::object& obj);
