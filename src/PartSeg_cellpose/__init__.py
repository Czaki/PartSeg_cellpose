
from .cellpose_plugin import CellposeSegmentation

def register():
    print("buka")
    from PartSegCore.register import register, RegisterEnum

    register(CellposeSegmentation, RegisterEnum.mask_algorithm)
