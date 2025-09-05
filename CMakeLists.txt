cmake_minimum_required(VERSION 3.10)
project(engine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)

# Add your source files
add_executable(${PROJECT_NAME}
    engine/main.cpp
    engine/networking.cpp
)

# Set the include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/external/asio/asio/include
    ${CMAKE_SOURCE_DIR}/external/msgpack-c/include
)

# Link the libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${CMAKE_SOURCE_DIR}/external/msgpack-c/lib/libmsgpackc.a
)
