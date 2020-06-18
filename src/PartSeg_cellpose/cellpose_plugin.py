from abc import ABC
from copy import deepcopy
from enum import Enum
from typing import Callable, List, Union, Tuple

import numpy as np

from PartSegCore.algorithm_describe_base import SegmentationProfile, AlgorithmProperty
from PartSegCore.channel_class import Channel
from PartSegCore.class_generator import enum_register
from PartSegCore.segmentation.algorithm_base import SegmentationResult, AdditionalLayerDescription
from PartSegCore.segmentation.segmentation_algorithm import StackAlgorithm

from cellpose import models


class CellposeModels(Enum):
    cyto = 0
    nuclei = 1

    def __str__(self):
        return self.name


enum_register.register_class(CellposeModels)


class CellPoseBase(StackAlgorithm, ABC):
    def __init__(self):
        super().__init__()
        self.parameters = {}
        self.model = models.Cellpose(gpu=False, model_type="cyto")

    @classmethod
    def get_fields(cls) -> List[Union[AlgorithmProperty, str]]:
        return [
            AlgorithmProperty("nucleus_channel", "Nucleus", 0, property_type=Channel),
            AlgorithmProperty("diameter", "Diameter", 30, options_range=(0, 10000)),
        ]

    def get_segmentation_profile(self) -> SegmentationProfile:
        return SegmentationProfile("", self.get_name(), deepcopy(self.parameters))

    def set_parameters(self, **kwargs):
        base_names = [x.name for x in self.get_fields() if isinstance(x, AlgorithmProperty)]
        if set(base_names) != set(kwargs.keys()):
            missed_arguments = ", ".join(set(base_names).difference(set(kwargs.keys())))
            additional_arguments = ", ".join(set(kwargs.keys()).difference(set(base_names)))
            raise ValueError(f"Missed arguments {missed_arguments}; Additional arguments: {additional_arguments}")
        self.parameters = deepcopy(kwargs)

    def get_data(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        raise NotImplementedError()

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> SegmentationResult:
        data, channels = self.get_data()
        masks, flows, _styles, _diams = self.model.eval(data, diameter=self.parameters["diameter"], channels=channels)
        add = {f"flows {i}": AdditionalLayerDescription(flow, layer_type="image") for i, flow in enumerate(flows)}
        add["mask"] = AdditionalLayerDescription(masks, layer_type="image")
        return SegmentationResult(masks, self.get_segmentation_profile(), additional_layers=add)


class CellposeCytoSegmentation(CellPoseBase):
    def get_data(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        nucleus_channel = np.squeeze(self.image.get_channel(self.parameters["nucleus_channel"]))
        cell_channel = np.squeeze(self.image.get_channel(self.parameters["cell_channel"]))
        channel_shape = nucleus_channel.shape + (1,)
        res = np.concatenate(
            [nucleus_channel.reshape(channel_shape), cell_channel.reshape(channel_shape)], axis=len(channel_shape) - 1
        )
        return res, (1, 2)

    def get_info_text(self):
        return ""

    @classmethod
    def get_name(cls):
        return "Cellpose cyto"

    @classmethod
    def get_fields(cls) -> List[Union[AlgorithmProperty, str]]:
        fields = super().get_fields()
        fields.insert(1, AlgorithmProperty("cell_channel", "Cells", 0, property_type=Channel))
        return fields


class CellposeNucleiSegmentation(CellPoseBase):
    def __init__(self):
        super().__init__()
        self.model = models.Cellpose(gpu=False, model_type="nuclei")

    def get_info_text(self):
        return ""

    @classmethod
    def get_name(cls):
        return "Cellpose nuclei"
