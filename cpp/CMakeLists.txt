cmake_minimum_required(VERSION 3.22)

project(stage0_producer_consumer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_library(pc_core STATIC
  src/pipeline.cpp
)
target_include_directories(pc_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(pc_core PUBLIC Threads::Threads)

add_executable(pc_demo src/main.cpp)
target_link_libraries(pc_demo PRIVATE pc_core)

include(CTest)
if(BUILD_TESTING)
  add_executable(pc_tests tests/test_bounded_queue.cpp)
  target_link_libraries(pc_tests PRIVATE pc_core)
  add_test(NAME bounded_queue_tests COMMAND pc_tests)
endif()

install(TARGETS pc_core pc_demo
  EXPORT pc_targets
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT pc_targets
  FILE pc_targets.cmake
  NAMESPACE stage0::
  DESTINATION lib/cmake/stage0
)
