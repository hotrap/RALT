cmake_minimum_required(VERSION 3.2)
project(viscnts)

set(CMAKE_CXX_FLAGS_MYREL "-Ofast -march=native -g")

IF (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()
add_compile_options(-Wall -Wextra -fexceptions)
IF (CMAKE_BUILD_TYPE STREQUAL Debug)
    ADD_DEFINITIONS(-DDEBUG)
ENDIF()

add_executable(test src/test.cpp src/test_splay.cpp src/viscnts.cpp src/bloom.cpp src/file.cpp src/hash.cpp src/memtable.cpp)
add_library(${PROJECT_NAME} SHARED src/viscnts.cpp src/bloom.cpp src/file.cpp src/hash.cpp src/memtable.cpp)

target_link_libraries(test PRIVATE pthread boost_fiber)
target_link_libraries(${PROJECT_NAME} PRIVATE pthread boost_fiber)


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/home/yfzcsc/hotrap/hotrap/include)

target_compile_features(test PRIVATE cxx_std_17)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
