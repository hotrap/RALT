cmake_minimum_required(VERSION 3.2)
project(viscnts)

set(CMAKE_CXX_FLAGS_MYREL "-march=native -g")

IF (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Relwithdebinfo" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()
add_executable(test src/test.cpp src/test_splay.cpp src/bloom.cpp src/file.cpp src/hash.cpp src/memtable.cpp)
add_library(${PROJECT_NAME} SHARED src/viscnts.cpp src/bloom.cpp src/file.cpp src/hash.cpp src/memtable.cpp)

add_subdirectory(third_party/fmt)

target_link_libraries(test PRIVATE pthread boost_fiber fmt)
target_link_libraries(${PROJECT_NAME} PRIVATE pthread boost_fiber fmt)


include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${ROCKSDB_INCLUDE})

target_compile_features(test PRIVATE cxx_std_17)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
