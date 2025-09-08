#pragma once
#include <string>
#include <unordered_map>
#include <msgpack.hpp>

enum class endpoints {
  start_session,
  end_session,
  step,
  update,
  save,
};

// Maps string paths to enum
static const std::unordered_map<std::string, endpoints> path_to_endpoint {
  {"/start_session", endpoints::start_session},
  {"/end_session", endpoints::end_session},
  {"/step", endpoints::step},
  {"/update", endpoints::update},
  {"/save", endpoints::save},
};

enum class http_method {
  post,
  get,
};

struct HttpRequest {
    http_method method {};        
    endpoints endpoint {};
    std::string http_version {};
    std::size_t content_length = 0;
    std::unordered_map<std::string, msgpack::object> body {};
};
