#pragma once
#include <pybind11/embed.h>
#include "http.h"

namespace py = pybind11;

extern std::unordered_map<std::string, py::object> model_map;
extern py::object torch_module;
extern py::object soft_ac_module;

void init_py();
void create_neural_network();
HttpResponse handle_start(const HttpRequest& req);
HttpResponse handle_end(const HttpRequest& req);
HttpResponse handle_step(const HttpRequest& req);
HttpResponse handle_update(const HttpRequest& req);

static const std::unordered_map<std::string, HttpResponse(*)(const HttpRequest&)> endpoint_to_handle {
  {"/start_session", handle_start},
  {"/end_session", handle_end},
  {"/step", handle_step},
  {"/update", handle_update},
};
