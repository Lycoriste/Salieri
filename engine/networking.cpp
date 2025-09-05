#define ASIO_STANDALONE
#include <asio.hpp>
#include <iostream>

using asio::ip::tcp;

class Session : public std::enable_shared_from_this<Session> {
  public:
    Session(tcp::socket socket) : socket_(std::move(socket)) {}
    void start() { do_read(); }

  private:
    tcp::socket socket_;
    enum { max_length = 1024 };
    char data_[max_length];

    void do_read() {
      auto self(shared_from_this());
      socket_.async_read_some(asio::buffer(data_, max_length),
      [this, self](std::error_code ec, std::size_t length) {
        if (!ec) {
          // Handle data
        } else if (ec == asio::error::eof) {
          std::cout << "[!] Client disconnected.\n";
        } else {
          std::cerr << "[!] Error: " << ec.message() << std::endl;
        }
      });
    }

    void do_write(std::size_t length) {
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

int networking() {
  asio::io_context io;
  Server server(io, 8080);
  io.run();

  return 0;
}
