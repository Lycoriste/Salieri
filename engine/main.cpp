#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include "handler.h"
#include "server.h"

namespace py = pybind11;

std::atomic<bool> running {true};

int main() {
  py::scoped_interpreter guard {false};
  init_py();

  std::signal(SIGINT, [](int){
      running = false;
      io.stop();
  });

  std::thread server_thread([] {
    run(8080);
  });

  while (running) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  server_thread.join();

  return 0;
}
