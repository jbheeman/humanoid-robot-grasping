# The existing bounded little-endian Z16/zstd envelope represented as ROS
# fields. checksum_sha256 covers the uncompressed pixels.
uint8 version
uint64 sequence
uint32 width
uint32 height
float32 depth_scale
float64 sensor_timestamp_ms
string timestamp_domain
string calibration_id
bool registered_to_rgb
string encoding
uint32 uncompressed_size
string checksum_sha256
uint8[] payload
