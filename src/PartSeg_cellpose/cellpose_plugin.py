from abc import ABC
from enum import Enum
from typing import Callable, List, Union

from PartSegCore.algorithm_describe_base import SegmentationProfile, AlgorithmProperty
from PartSegCore.channel_class import Channel
from PartSegCore.segmentation.algorithm_base import SegmentationResult
from PartSegCore.segmentation.segmentation_algorithm import StackAlgorithm

from cellpose import models


class CellposeModels(Enum):
    cyto=0
    nuclei=1

    def __str__(self):
        return self.name

class CellPoseBase(StackAlgorithm, ABC):
    def __init__(self):
        super().__init__()
        self.model = models.Cellpose()

class CellposeCytoSegmentation(StackAlgorithm):
    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        pass

    def get_info_text(self):
        return ""

    def set_parameters(self, **kwargs):
        pass

    def get_segmentation_profile(self) -> SegmentationProfile:
        pass

    @classmethod
    def get_name(cls):
        return "Cellpose cyto"

    @classmethod
    def get_fields(cls) -> List[Union[AlgorithmProperty, str]]:
        return [
            AlgorithmProperty("cell_channel", "Cells", 0, property_type=Channel),
            AlgorithmProperty("nucleus_channel", "Nucleus", 0, property_type=Channel),
            AlgorithmProperty("diameter", "Diameter", 30, options_range=(0, 10000))
        ]


class CellposeNucleiSegmentation(StackAlgorithm):
    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        pass

    def get_info_text(self):
        return ""

    def set_parameters(self, **kwargs):
        pass

    def get_segmentation_profile(self) -> SegmentationProfile:
        pass

    @classmethod
    def get_name(cls):
        return "Cellpose cyto"

    @classmethod
    def get_fields(cls) -> List[Union[AlgorithmProperty, str]]:
        return [
            AlgorithmProperty("cell_channel", "Cells", 0, property_type=Channel),
            AlgorithmProperty("diameter", "Diameter", 30, options_range=(0, 10000))
        ]