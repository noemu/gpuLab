cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(gpuLab1)


#include_directories(lib)

#add_definitions(-DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB) #not sure why i should need this
set(Boost_USE_STATIC_LIBS ON) #boost standard compilation is static (.lib) not dynamic(.dll)
set(Boost_USE_MULTITHREADED ON)

#SET(BOOST_ROOT "D:/Librarys/boost")
#set(BOOST_LIBRARYDIR "D:/librarys/boost/stage/lib")
set(Boost_DEBUG 0) #debug option

LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

find_package(OpenCL REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)

#find dbghelp
if(MSVC AND NOT DEFINED DBGHELP_INCLUDE_DIR AND NOT DEFINED DBGHELP_LIBRARY)
	set(DBGHELP_INCLUDE_DIR "")
	set(DBGHELP_LIBRARY "dbghelp")
	set(DBGHELP_FOUND TRUE)
else()
set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")
endif()


# Directories
set(GPULAB_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GPULAB_SRC_DIR ${GPULAB_DIR}/src)
set(GPULAB_LIB_DIR ${GPULAB_DIR}/lib)


file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
file(GLOB_RECURSE client_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cl")
file(GLOB_RECURSE lib_source_files_CPP RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "lib/*.cpp")
file(GLOB_RECURSE lib_source_files_C RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "lib/*.c")


# Grouping in Visual Studio #not working atm
source_group("Public Header Files" FILES ${public_header_files})
source_group("Header Files" FILES ${header_files})
source_group("Source Files" FILES ${source_files})
source_group("Client Files" FILES ${client_files})

#some status messages
message(STATUS "\n-------------------------------------")
message(STATUS "Found OpenCL Include Dir: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "Found OpenCL Library Dir: ${OpenCL_LIBRARY}")
message(STATUS "-------------------------------------")
message(STATUS "Found BOOST Include Dir: ${Boost_INCLUDE_DIRS}")
message(STATUS "Found BOOST Library Dir: ${Boost_FILESYSTEM_LIBRARY}")
message(STATUS "-------------------------------------")
message(STATUS "Found Windows DEBUG Library: ${DBGHELP_LIBRARY}")
message(STATUS "-------------------------------------\n")


#include_directories(${client_files})
include_directories(${OpenCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${GPULAB_LIB_DIR}) #header files
add_executable (${PROJECT_NAME} ${source_files} ${public_header_files} ${header_files} ${lib_source_files_CPP} ${lib_source_files_C} ${client_files}) #files to compile/add to solution
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON) #idk
target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${OpenCL_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} ${DBGHELP_LIBRARY}) #link static/dynamic libraries
set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${GPULAB_DIR}") #set correct relative path to debugger, to find the cl files
