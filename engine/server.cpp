#include "net_common.h"

using asio::ip::tcp;

class Session : public std::enable_shared_from_this<Session> {
  public:
    Session(tcp::socket socket) : socket_(std::move(socket)) {}
    void start() { do_read_header(); }

  private:
    tcp::socket socket_;
    std::string buffer_ {};

    void do_read_header() {
      auto self(shared_from_this());
      asio::async_read_until(socket_, asio::dynamic_buffer(buffer_), "\r\n\r\n",
      [this, self](std::error_code ec, std::size_t bytes_transferred) {
        if (!ec) {
          std::string headers = buffer_.substr(0, bytes_transferred);
          buffer_.erase(0, bytes_transferred);

          std::cout << "\n========HEADER=========\n" << headers << std::endl;

          std::istringstream stream(headers);
          std::string request_line;
          std::getline(stream, request_line); // Skip the "POST / HTTP/1.1"

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

          if (content_length > 0) {
            // buffer_.resize(content_length);
            do_read_body(content_length);
          }
        }
      });
    }

    void do_read_body(size_t content_length) {
      auto self(shared_from_this());

      if (buffer_.size() >= content_length) {
        handle_body(content_length);
        return;
      }

      size_t data_read = buffer_.size();
      buffer_.resize(content_length);

      asio::async_read(socket_, asio::buffer(buffer_.data() + data_read, content_length - data_read),
      [this, self, content_length](std::error_code ec, std::size_t length) {
        if (!ec) {
          handle_body(content_length);
        } else {
          std::cerr << "[!] Error reading header: " << ec.message() << std::endl;
        }
      });
    }

    void handle_body(size_t content_length) {
      std::cout << "[DEBUG] Received body (" << content_length << " bytes)\n";

      try {
        msgpack::object_handle oh = msgpack::unpack(buffer_.data(), buffer_.size());
        msgpack::object obj = oh.get();

        std::map<std::string, std::string> decoded;
        obj.convert(decoded);

        std::cout << "[+] Payload: " << decoded["payload"] << std::endl;
        send_response();
      } catch (const std::exception& e) {
        std::cerr << "[!] Msgpack error: " << e.what() << std::endl;
      }
    }

    void send_response() {
      auto self(shared_from_this());
      const std::string resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK";
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
  asio::io_context io;
  Server server(io, 8080);
  io.run();

  return 0;
}
