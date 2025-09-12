#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include "handler.h"
#include "server.h"

namespace py = pybind11;

std::atomic<bool> running {true};

void signal_handler(int sig) {
      running = false;
      std::cout.flush();
      io.stop();
}

int main() {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  std::thread server_thread([] {
    py::scoped_interpreter guard{true};
    init_py();
    run(8080);
  });

  while (running) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  server_thread.join();

  return 0;
}
