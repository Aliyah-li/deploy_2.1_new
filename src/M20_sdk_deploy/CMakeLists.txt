cmake_minimum_required(VERSION 3.10)

project(m20_sdk_deploy)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认构建类型
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# 警告忽略和编译选项
add_definitions(-w)
add_compile_options(-fPIC)

# 配置选项
set(BUILD_PLATFORM "x86" CACHE STRING "Target platform (x86 or arm)")

if(BUILD_PLATFORM STREQUAL "arm")
  set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
  set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++") 
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(drdds REQUIRED)
find_package(std_msgs REQUIRED)

# Evdev keyboard support is intentionally disabled for portable ARM/no-lib builds.
# Keep this block for rollback if direct /dev/input keyboard input is needed again.
# include(CheckIncludeFileCXX)
# check_include_file_cxx("libevdev-1.0/libevdev/libevdev.h" HAVE_LIBEVDEV_HEADER)
# find_library(EVDEV_LIBRARY NAMES evdev libevdev.so.2 PATHS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)

set(THIRD_PARTY ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(DEFINED ENV{CONDA_PREFIX})
  link_directories($ENV{CONDA_PREFIX}/lib)
endif()

include_directories(
  include/types
  include/utils
  interface/robot
  interface/user_command
  state_machine
  run_policy
  ${THIRD_PARTY}/eigen/
  ${THIRD_PARTY}/gamepad/include
  ${THIRD_PARTY}/onnxruntime/${BUILD_PLATFORM}/include
)

# 子目录
add_subdirectory(interface)

# 运行时路径设置
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


set(CMAKE_INSTALL_RPATH 
  ${THIRD_PARTY}/onnxruntime/${BUILD_PLATFORM}/lib/
  ../lib/onnxruntime/
)

file(GLOB_RECURSE STATE_MACHINE_SRC "state_machine/*.c*")

add_executable(rl_deploy main.cpp ${STATE_MACHINE_SRC})

# ── FFT sim2sim executable ──
add_executable(rl_deploy_fft main_fft.cpp ${STATE_MACHINE_SRC})

# ── FFT2 sim2sim executable (变高度) ──
add_executable(rl_deploy_fft2 main_fft2.cpp ${STATE_MACHINE_SRC})

# ── Strong2 sim2sim executable (变高度) ──
add_executable(rl_deploy_strong2 main_strong2.cpp ${STATE_MACHINE_SRC})

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${THIRD_PARTY}/onnxruntime//${BUILD_PLATFORM}/lib/libonnxruntime.so")

ament_target_dependencies(rl_deploy
  rclcpp
  drdds
  std_msgs
)
ament_target_dependencies(rl_deploy_fft
  rclcpp
  drdds
  std_msgs
)
ament_target_dependencies(rl_deploy_fft2
  rclcpp
  drdds
  std_msgs
)
ament_target_dependencies(rl_deploy_strong2
  rclcpp
  drdds
  std_msgs
)
if(BUILD_PLATFORM STREQUAL "arm")
  target_link_libraries(rl_deploy
    interface
    onnxruntime
    -lpthread
    -lm
    -lrt
    -ldl
    -lstdc++
  )
  target_link_libraries(rl_deploy_fft
    interface
    onnxruntime
    -lpthread
    -lm
    -lrt
    -ldl
    -lstdc++
  )
  target_link_libraries(rl_deploy_fft2
    interface
    onnxruntime
    -lpthread
    -lm
    -lrt
    -ldl
    -lstdc++
  )
  target_link_libraries(rl_deploy_strong2
    interface
    onnxruntime
    -lpthread
    -lm
    -lrt
    -ldl
    -lstdc++
  )
else()
  target_link_libraries(rl_deploy
    interface
    onnxruntime
    -lpthread
    -lm
    -lrt
    -ldl
    -lstdc++
  )
  target_link_libraries(rl_deploy_fft
    interface
    onnxruntime
    -lpthread
    -lm
    -lrt
    -ldl
    -lstdc++
  )
  target_link_libraries(rl_deploy_fft2
    interface
    onnxruntime
    -lpthread
    -lm
    -lrt
    -ldl
    -lstdc++
  )
  target_link_libraries(rl_deploy_strong2
    interface
    onnxruntime
    -lpthread
    -lm
    -lrt
    -ldl
    -lstdc++
  )
endif()


install(TARGETS rl_deploy rl_deploy_fft rl_deploy_fft2 rl_deploy_strong2
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  config/teleop_config.json
  config/mixed_terrains_history_vae_config.json
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config/terrain/
  DESTINATION lib/${PROJECT_NAME}/terrain
  FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY policy/
  DESTINATION lib/${PROJECT_NAME}/policy
  FILES_MATCHING PATTERN "*.onnx"
)

ament_package()
