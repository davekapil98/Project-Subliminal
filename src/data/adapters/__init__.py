"""Source-specific public-dataset adapters."""

from data.adapters.armnetbench_so101 import (
    ArmnetBenchSO101Adapter,
    ArmnetBenchSourceSpec,
    QualifiedObject,
)
from data.adapters.droid_raw import (
    DROIDCameraSpec,
    DROIDObject,
    DROIDRawAdapter,
    DROIDSourceSpec,
)
from data.adapters.project_ira_so101 import (
    CameraSpec,
    ProjectIRASO101Adapter,
    ProjectIRASourceSpec,
    QualifiedFile,
    VideoSegment,
)
from data.adapters.so101_ma_multitask_700 import (
    SO101MAMultiTaskAdapter,
    SO101MAMultiTaskSourceSpec,
    UpstreamSource,
)

__all__ = [
    "ArmnetBenchSO101Adapter",
    "ArmnetBenchSourceSpec",
    "QualifiedObject",
    "DROIDCameraSpec",
    "DROIDObject",
    "DROIDRawAdapter",
    "DROIDSourceSpec",
    "CameraSpec",
    "ProjectIRASO101Adapter",
    "ProjectIRASourceSpec",
    "QualifiedFile",
    "VideoSegment",
    "SO101MAMultiTaskAdapter",
    "SO101MAMultiTaskSourceSpec",
    "UpstreamSource",
]
