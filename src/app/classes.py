"""
    Object definitions utilized by our resource endpoints.
"""

from enum import Enum

class ImagesType(Enum):
    """
        Enum that contains all the possible classes of the CIFAR 10 dataset
    """
    AIRPLANE = 0
    AUTOMOBILE = 1
    BIRD = 2
    CAT = 3
    DEER = 4
    DOG = 5
    FROG = 6
    HORSE = 7
    SHIP = 8
    TRUCK = 9