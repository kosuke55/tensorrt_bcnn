cmake_minimum_required(VERSION 2.8.3)
project(tensorrt_bcnn)

# set flags for CUDA availability
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDA)
if (CUDA_FOUND)
  message("CUDA is available!")
  message("CUDA Libs: ${CUDA_LIBRARIES}")
  message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  set(CUDA_AVAIL ON)
else()
  message("CUDA NOT FOUND")
  set(CUDA_AVAIL OFF)
endif (CUDA_FOUND)

# set flags for TensorRT availability
option(TRT_AVAIL "TensorRT available" OFF)
# try to find the tensorRT modules
find_library(NVINFER NAMES nvinfer)
find_library(NVPARSERS NAMES nvparsers)
find_library(NVCAFFE_PARSER NAMES nvcaffe_parser)
find_library(NVINFER_PLUGIN NAMES nvinfer_plugin)
if(NVINFER AND NVPARSERS AND NVCAFFE_PARSER AND NVINFER_PLUGIN)
   message("TensorRT is available!")
   message("NVINFER: ${NVINFER}")
   message("NVPARSERS: ${NVPARSERS}")
   message("NVCAFFE_PARSER: ${NVCAFFE_PARSER}")
   set(TRT_AVAIL ON)
else()
  message("TensorRT is NOT Available")
  set(TRT_AVAIL OFF)
endif()

# set flags for CUDNN availability
option(CUDNN_AVAIL "CUDNN available" OFF)
# try to find the CUDNN module
find_library(CUDNN_LIBRARY
NAMES libcudnn.so${__cudnn_ver_suffix} libcudnn${__cudnn_ver_suffix}.dylib ${__cudnn_lib_win_name}
PATHS $ENV{LD_LIBRARY_PATH} ${__libpath_cudart} ${CUDNN_ROOT_DIR} ${PC_CUDNN_LIBRARY_DIRS} ${CMAKE_INSTALL_PREFIX}
PATH_SUFFIXES lib lib64 bin
DOC "CUDNN library." )
if(CUDNN_LIBRARY)
   message("CUDNN is available!")
   message("CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
   set(CUDNN_AVAIL ON)
else()
  message("CUDNN is NOT Available")
  set(CUDNN_AVAIL OFF)
endif()

if(TRT_AVAIL AND CUDA_AVAIL AND CUDNN_AVAIL)

  find_package(catkin REQUIRED COMPONENTS
   roscpp
   roslib
   pcl_ros
   autoware_msgs
   cv_bridge
   )
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

  catkin_package(
   CATKIN_DEPENDS
   roscpp
   roslib
   pcl_ros
   autoware_msgs
   cv_bridge
   )

  include_directories(
   include
   lib/include
   ${catkin_INCLUDE_DIRS}
  )
  set(SOURCE_FILES
   src/tensorrt_bcnn_main.cpp
   src/tensorrt_bcnn_ros.cpp
   src/feature_generator.cpp
   )

  add_executable(tensorrt_bcnn
         ${SOURCE_FILES})

  add_dependencies(tensorrt_bcnn
         ${catkin_EXPORTED_TARGETS}
         )


  cuda_add_library(gpu_tensorrt_bcnn_lib
         lib/src/UpsampleLayer.cu
         )

  target_link_libraries(gpu_tensorrt_bcnn_lib
   ${CUDA_LIBRARIES}
  )

  add_library(tensorrt_bcnn_lib
         lib/src/TrtNet.cpp
         lib/src/UpsampleLayer.cpp
         )

  target_link_libraries(tensorrt_bcnn_lib
         ${NVINFER}
         ${NVCAFFE_PARSER}
         ${NVINFER_PLUGIN}
         ${CUDA_LIBRARIES}
         ${CUDA_CUBLAS_LIBRARIES}
         ${CUDA_curand_LIBRARY}
         ${CUDNN_LIBRARY}
         gpu_tensorrt_bcnn_lib
         )

  target_link_libraries(tensorrt_bcnn
         ${catkin_LIBRARIES}
         tensorrt_bcnn_lib
         )


  install(TARGETS
         gpu_tensorrt_bcnn_lib
         tensorrt_bcnn_lib
         tensorrt_bcnn
         ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
         LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
         RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

  install(DIRECTORY launch/
         DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
         PATTERN ".svn" EXCLUDE)

  install(DIRECTORY include/
         DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}/${PROJECT_NAME}/
         PATTERN ".svn" EXCLUDE
         )

else()
  find_package(catkin REQUIRED)
  catkin_package()
endif()
