cmake_minimum_required(VERSION 3.10)
project(RangeView)

set(CMAKE_CXX_STANDARD 17)
set(ONNXRUNTIME_DIR "/workspace/onnxruntime")

find_package(PCL 1.3 REQUIRED)
find_package(yaml-cpp  REQUIRED)
find_package(Torch REQUIRED PATHS "/workspace/libtorch")

include_directories(include)
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(rangenet src/EXnet.cpp)

target_link_libraries(rangenet ${PCL_LIBRARIES})
target_link_libraries(rangenet yaml-cpp)
target_link_libraries(rangenet ${TORCH_LIBRARIES})
target_compile_features(rangenet PUBLIC cxx_std_17)
target_link_libraries(rangenet onnxruntime pthread)