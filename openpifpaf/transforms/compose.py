from typing import List
import torch
from .preprocess import Preprocess, Preprocess_FM


class Compose(Preprocess):
    """Execute given transforms in sequential order."""
    def __init__(self, preprocess_list: List[Preprocess]):
        """

        :rtype: object
        """
        self.preprocess_list = preprocess_list

    def __call__(self, *args):
    # *args allows you to pass a variable number of arguments to the method, img and anns here
        for p in self.preprocess_list:  # each preprocess methods
            if p is None:
                continue
            args = p(*args)  #  the output of each preprocessing function will be passed as input to the next one in the list.
        return args  # results of final preprocess function


class Compose_FM(Preprocess_FM):
    """Execute given transforms in sequential order."""
    def __init__(self, preprocess_list: List[Preprocess_FM]):
        """

        :rtype: object
        """
        self.preprocess_list = preprocess_list

    def __call__(self, *args):
    # *args allows you to pass a variable number of arguments to the method, img and anns here
        for p in self.preprocess_list:  # each preprocess methods
            if p is None:
                continue
            if not isinstance(args, tuple):
                args = (args,)
            # input *args to unpack
            args = p(*args)  #  the output of each preprocessing function will be passed as input to the next one in the list.
            # if the input *args here is a single tensor instead of a tuple, the *args operation would wrongly unpack the tensor
            if not isinstance(args, tuple):
                args = (args,)

        return args  # results of final preprocess function
