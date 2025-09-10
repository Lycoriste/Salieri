#pragma once
#include <string>
#include <unordered_map>
#include <msgpack.hpp>

enum class HttpMethod {
  Post,
  Get,
};

enum class HttpStatusCode {
    Ok = 200,
    Created = 201,
    BadRequest = 400,
    NotFound = 404,
    InternalServerError = 500
};

struct HttpRequest {
    HttpMethod method {};        
    std::string endpoint {};
    std::string http_version {};
    std::size_t content_length = 0;
    msgpack::object body_view {};
    msgpack::object_handle body_handle {};
};

struct HttpResponse {
  std::string http_version {};
  std::string status_code {};
  std::size_t content_length {};
  msgpack::object content {};
};
