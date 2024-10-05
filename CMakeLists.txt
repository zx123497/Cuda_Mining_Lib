cmake_minimum_required(VERSION 3.10)
project(cuda_mining_lib LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable CUDA
enable_language(CUDA)

# Add the library target
add_library(mining_lib STATIC src/mining.cu)

# Include directories for headers
target_include_directories(mining_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link necessary libraries (if needed)
# target_link_libraries(mining_lib some_other_library)

