cmake_minimum_required(VERSION 3.10)
project(engine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Add your source files
add_executable(${PROJECT_NAME}
    engine/main.cpp
    engine/server.cpp
    engine/handler.cpp
)

# Set the include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/external/asio/asio/include
    ${CMAKE_SOURCE_DIR}/external/msgpack-c/include
)

# Link the libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
  pybind11::embed
  Python3::Python
)
