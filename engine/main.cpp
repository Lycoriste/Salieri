#include <iostream>
#include "handler.h"
#include "server.h"

namespace py = pybind11;

int main() {
  py::scoped_interpreter guard {};
  init_py();
  run();
  return 1;
}
