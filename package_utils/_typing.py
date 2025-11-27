#-*- coding: utf-8 -*-
from typing import Dict, List, Optional, Tuple, Union

from mmengine.config import ConfigDict
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]

# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

# Type hint of data samples
InstanceList = List[InstanceData]
PixelDataList = List[PixelData]
Predictions = Union[InstanceList, Tuple[InstanceList, PixelDataList]]

# Type hint of features
Features = Union[Tuple[Tensor], List[Tuple[Tensor]], List[List[Tuple[Tensor]]]]
