from typing import List

from .preprocess import Preprocess


class Compose(Preprocess):
    """Execute given transforms in sequential order."""
    def __init__(self, preprocess_list: List[Preprocess]):
        self.preprocess_list = preprocess_list

    def __call__(self, *args):
    # *args allows you to pass a variable number of arguments to the method, img and anns here
        for p in self.preprocess_list:  # each preprocess methods
            if p is None:
                continue
            args = p(*args)  #  the output of each preprocessing function will be passed as input to the next one in the list.

        return args  # results of final preprocess function
