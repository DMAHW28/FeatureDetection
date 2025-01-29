cmake_minimum_required(VERSION 3.29)
project(FeatureDetection)

set(CMAKE_CXX_STANDARD 14)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable first
add_executable(FeatureDetection
        cpp/src/main.cpp
        cpp/src/harris.cpp
        cpp/src/fast.cpp
        cpp/src/descriptors.cpp
)

# Link OpenCV libraries to the executable
target_link_libraries(FeatureDetection ${OpenCV_LIBS})

# Print debug messages
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
