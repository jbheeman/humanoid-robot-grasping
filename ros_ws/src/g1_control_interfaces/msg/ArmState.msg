# Typed safety summary. report_json carries the complete bridge report so new
# fields can be added without breaking Foxy/Jazzy wire compatibility.
bool ok
string state
string session_id
string control_mode
string calibration_id
int64 last_sequence
uint32 last_target_age_ms
uint32 robot_state_age_ms
float64[14] measured_arm_q
float64[14] commanded_arm_q
float32 weight
bool robot_state_fresh
bool standing
bool compatible_motion_mode
bool controller_available
bool motion_mode_verified
bool controller_ownership_verified
bool motor_status_verified
bool motor_state_healthy
string[] motor_faults
string fault_reason
string hold_reason
string report_json
