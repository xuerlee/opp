"""Collections of decoders: fields to annotations."""

from openpifpaf.decoder import utils
from openpifpaf.decoder.decoder import Decoder
from openpifpaf.decoder.cifcaf import CifCaf
from openpifpaf.decoder.cifdet import CifDet
from openpifpaf.decoder.pose_similarity import PoseSimilarity
from openpifpaf.decoder.tracking_pose import TrackingPose
from openpifpaf.decoder.factory import cli, configure, factory

from openpifpaf.decoder.factory import DECODERS
