cmake_minimum_required( VERSION 3.0 )

project(cvyolo)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB YOLO_SOURCE "source/*.cpp")
file(GLOB YOLO_HEADER "include/*.h")

add_executable(yolo ${YOLO_SOURCE})
target_link_libraries(yolo ${OpenCV_LIBS})
