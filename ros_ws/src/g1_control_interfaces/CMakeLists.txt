cmake_minimum_required(VERSION 3.5)
project(g1_control_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArmTarget.msg"
  "msg/ArmState.msg"
  "msg/CompressedDepth.msg"
  "msg/CommissioningState.msg"
  "srv/ArmControl.srv"
  "srv/CommissioningCommand.srv"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
