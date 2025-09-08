#pragma once

#include <iostream>
#include <string>
#include <thread>
#include <memory>
#include <optional>
#include <mutex>
#include <deque>
#include <vector>
#include <algorithm>
#include <chrono>
#include <pybind11/embed.h>

#ifdef _WIN32
#define _WIN32_WINNT 0x0A00
#endif

#define ASIO_STANDALONE
#include <asio.hpp>
#include <asio/ts/buffer.hpp>
#include <asio/ts/internet.hpp>

#define MSGPACK_NO_BOOST
#include <msgpack.hpp>
