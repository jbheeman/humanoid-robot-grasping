# Absolute target for exactly one seven-joint G1 arm. The robot-local bridge
# merges left and right targets into the single 14-joint Unitree arm frame.
std_msgs/Header header
string session_id
uint64 sequence
string[7] joint_names
float64[7] position_rad
builtin_interfaces/Duration move_duration
