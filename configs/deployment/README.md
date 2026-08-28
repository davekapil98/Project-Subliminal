# Deployment configurations

Physical safety, runtime rates, reference precision and optional local
quantization exceptions belong here after reference checkpoints exist.

The v1.3 baseline uses FP16/FP32 reference metrics and may evaluate INT8 or
weight-only 4-bit deployment only when accuracy gates pass. NVFP4 is an
optional future Blackwell export path, not a mandatory configuration target.
