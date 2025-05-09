cmake_minimum_required(VERSION 3.2)
project(ralt)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_MYREL "-march=native -g")

IF (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Relwithdebinfo" CACHE STRING
		"Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

option(WITH_ASAN "build with ASAN" OFF)
if(WITH_ASAN)
	add_compile_options(-fsanitize=address)
endif()

IF (NOT_DIRECTIO)
	add_definitions(-DNOT_DIRECTIO)
ENDIF()
IF (USE_LRU)
	add_definitions(-DUSE_LRU)
ENDIF()
IF (USE_CLOCK)
	add_definitions(-DUSE_CLOCK)
ENDIF()
add_executable(test_basic src/test_basic.cpp src/test_splay.cpp src/file.cpp src/hash.cpp)
add_executable(test_viscnts src/viscnts.cpp src/test_viscnts.cpp src/file.cpp src/hash.cpp)
add_library(${PROJECT_NAME} SHARED src/viscnts.cpp src/file.cpp src/hash.cpp)

target_link_libraries(test_basic PRIVATE pthread fmt)
target_link_libraries(test_viscnts PRIVATE pthread fmt uring rocksdb)
target_link_libraries(${PROJECT_NAME} PRIVATE pthread fmt)
target_link_directories(test_viscnts PRIVATE ${ROCKSDB_LIB})

include_directories(${PROJECT_SOURCE_DIR}/include ${ROCKSDB_INCLUDE})
