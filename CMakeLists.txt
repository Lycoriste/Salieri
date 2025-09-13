cmake_minimum_required(VERSION 3.16)
project(engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)

FetchContent_Declare(
  asio
  GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
  GIT_TAG        asio-1-36-0
  SOURCE_SUBDIR  asio/include
)

FetchContent_Declare(
  msgpack-c
  GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
  GIT_TAG        a0b2ec09da4bd823e40fa591221713951d4ec995
  SOURCE_SUBDIR  include
)

FetchContent_MakeAvailable(pybind11)
FetchContent_MakeAvailable(asio)
FetchContent_MakeAvailable(msgpack-c)

set(ASIO_INCLUDE_DIR ${asio_SOURCE_DIR}/asio/include)
set(MSGPACK_INCLUDE_DIR ${msgpack-c_SOURCE_DIR}/include)

# Add your source files
add_executable(${PROJECT_NAME}
    engine/main.cpp
    engine/server.cpp
    engine/handler.cpp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    ASIO_STANDALONE          # Use standalone ASIO (no BOOST)
    ASIO_NO_DEPRECATED       # Disable deprecated features
    MSGPACK_NO_BOOST
)

# Set the include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${ASIO_INCLUDE_DIR}
    ${MSGPACK_INCLUDE_DIR}
)

# Link the libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    pybind11::embed
    Python3::Python
)
