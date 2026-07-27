cmake_minimum_required(VERSION 3.5)
project(g1_control_interfaces)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArmManualRequest.msg"
  "msg/ArmManualResponse.msg"
  "msg/ArmSideTarget.msg"
  "msg/ArmTarget.msg"
  "msg/ArmState.msg"
  "msg/CompressedDepth.msg"
  "msg/CommissioningState.msg"
  "msg/CommissioningRequest.msg"
  "msg/CommissioningResponse.msg"
  "srv/ArmControl.srv"
  "srv/CommissioningCommand.srv"
  DEPENDENCIES builtin_interfaces std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
