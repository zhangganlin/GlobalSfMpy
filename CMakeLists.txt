project(GlobalSfMpy)
cmake_minimum_required(VERSION 3.4...3.18)
set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC" )
# set(CMAKE_CXX_STANDARD 17)

find_package(Theia REQUIRED)
find_package(pybind11 REQUIRED)
find_package( OpenCV REQUIRED )
include_directories(${THEIA_INCLUDE_DIRS})
include_directories(${GlobalSfMpy_SOURCE_DIR}/include)
include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB_RECURSE SRC_GLOBALSFMPY
	"src/*.cpp"
)

find_package(pybind11 REQUIRED)
pybind11_add_module(
    GlobalSfMpy 
    bind_src/GlobalSfMpy.cpp
    ${SRC_GLOBALSFMPY}
)

target_link_libraries(
    GlobalSfMpy PRIVATE 
    ${THEIA_LIBRARIES}
    ${OpenCV_LIBS}
)
