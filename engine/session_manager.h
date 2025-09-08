#pragma once
#include <string>
#include <mutex>
#include <unordered_map>

// More metadata?
struct SessionInfo {
  std::chrono::steady_clock::time_point last_auth;
};

class SessionManager {
  public:
    static SessionManager& instance() {
      static SessionManager mgr;
      return mgr;
    };

  private:
    std::unordered_map<int, SessionInfo> sessions_;
    std::mutex mutex_;
};
