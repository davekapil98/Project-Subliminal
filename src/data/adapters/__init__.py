"""Source-specific public-dataset adapters."""

from data.adapters.armnetbench_so101 import (
    ArmnetBenchSO101Adapter,
    ArmnetBenchSourceSpec,
    QualifiedObject,
)
from data.adapters.project_ira_so101 import (
    CameraSpec,
    ProjectIRASO101Adapter,
    ProjectIRASourceSpec,
    QualifiedFile,
    VideoSegment,
)

__all__ = [
    "ArmnetBenchSO101Adapter",
    "ArmnetBenchSourceSpec",
    "QualifiedObject",
    "CameraSpec",
    "ProjectIRASO101Adapter",
    "ProjectIRASourceSpec",
    "QualifiedFile",
    "VideoSegment",
]
