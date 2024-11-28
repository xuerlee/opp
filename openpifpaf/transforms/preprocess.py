from abc import ABCMeta, abstractmethod


class Preprocess(metaclass=ABCMeta):
    """Preprocess an image with annotations and meta information."""
    @abstractmethod
    def __call__(self, image, anns, meta):
        """Implementation of preprocess operation."""

class Preprocess_FM(metaclass=ABCMeta):
    # the constraint and requirements of the subclass
    """Preprocess an image"""
    @abstractmethod  # the subclass must have this method
    def __call__(self, image):
        """Implementation of preprocess operation."""
