#include "net_common.h"
#include "http.h"
#include "handler.h"
#include "mputil.h"
#include "session_manager.h"

using asio::ip::tcp;

asio::io_context io;
std::unordered_set<int> active_sessions;
std::mutex session_mutex;

class Session : public std::enable_shared_from_this<Session> {
  public:
    Session(tcp::socket socket) : socket_(std::move(socket)) {}
    void start() { do_read_header(); }

  private:
    tcp::socket socket_;
    std::string buffer_ {};
    HttpRequest req_ {};

    void do_read_header() {
      auto self(shared_from_this());
      asio::async_read_until(socket_, asio::dynamic_buffer(buffer_), "\r\n\r\n",
      [this, self](std::error_code ec, std::size_t bytes_transferred) {
        if (!ec) {
          std::string headers = buffer_.substr(0, bytes_transferred);
          buffer_.erase(0, bytes_transferred);

          std::istringstream stream(headers);
          std::string request_line;
          std::getline(stream, request_line); // Skip the "POST / HTTP/1.1"
          
          if (!request_line.empty() && request_line.back() == '\r') 
            request_line.pop_back();
          
          std::istringstream rq_stream(request_line);
          std::string method, path;
          rq_stream >> method >> req_.endpoint;
          req_.method = method == "POST" ? HttpMethod::Post : HttpMethod::Get;  // I don't care

          std::string line {};
          std::size_t content_length = 0;

          // TODO: Add authorization
          while (std::getline(stream, line)) {
            if (line.back() == '\r') line.pop_back();
            if (line.empty()) break;

            if (line.find("Content-Length:") == 0) {
              std::string len_str = line.substr(15);
              len_str.erase(0, len_str.find_first_not_of("\t"));
              content_length = std::stoul(len_str);
            }
          }

          if (content_length <= 0) return;
          req_.content_length = content_length;
          do_read_body();
        }
      });
    }

    void do_read_body() {
      auto self(shared_from_this());

      if (buffer_.size() >= req_.content_length) {
        handle_body();
        return;
      }

      size_t data_read = buffer_.size();
      buffer_.resize(req_.content_length);

      asio::async_read(socket_, asio::buffer(buffer_.data() + data_read, req_.content_length - data_read),
      [this, self](std::error_code ec, std::size_t length) {
        if (!ec) {
          handle_body();
        } else {
          std::cerr << "[!] Error reading body: " << ec.message() << std::endl;
        }
      });
    }
    
    // Reads data and dispatches to endpoints
    void handle_body() {
      try {
        req_.body_handle = msgpack::unpack(buffer_.data(), buffer_.size());
        req_.body_view = req_.body_handle.get();

        if (req_.body_view.type != msgpack::type::MAP) {
            std::cerr << "[!] Invalid message format\n";
            return;
        }

        if (auto server_id_opt = get_field<int>(req_.body_view, "server_id")) {
            int server_id = *server_id_opt;
            std::cout << "server_id: " << server_id << std::endl;
        } else {
            std::cerr << "server_id not found or not an int!" << std::endl;
        }

        auto it = endpoint_to_handle.find(req_.endpoint);
        if (it != endpoint_to_handle.end()) {
          it->second(req_);
        } else {
          std::cerr << "[!] Failed to find endpoint.\n";
        }

      } catch (const std::exception& e) {
        std::cerr << "[!] Error handling request content: " << e.what() << std::endl;
        std::cerr << " └── Endpoint: " << req_.endpoint << std::endl;
      }
    }

    void send_response(const std::string& resp) {
      auto self(shared_from_this());
      asio::async_write(socket_, asio::buffer(resp),
      [this, self](std::error_code ec, std::size_t length) {
        if (!ec) {
          socket_.close();
        }
      });
    }
};

class Server {
  public:
    Server(asio::io_context& io, short port) : acceptor_(io, tcp::endpoint(tcp::v4(), port)) {
      std::cout << "[+] Server listening on port " << port << std::endl;
      do_accept();
    }

  private:
    tcp::acceptor acceptor_;

    void do_accept() {
      acceptor_.async_accept([this](std::error_code ec, tcp::socket socket) {
        if (!ec) {
          std::make_shared<Session>(std::move(socket))->start();
          std::cout << "[+] Client connected\n";
        }
        do_accept();
      });
    }
};

int run() {
  Server server(io, 8080);
  io.run();

  return 0;
}
