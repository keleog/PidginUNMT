# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
from collections import OrderedDict
from logging import getLogger
import numpy as np
import torch
from torch import nn
logger = getLogger()
from src.evaluator import EvaluatorMT


TOOLS_PATH = '../tools'
BLEU_SCRIPT_PATH = os.path.join(TOOLS_PATH, 'mosesdecoder/scripts/generic/multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH), "Moses not found. Please be sure you downloaded Moses in %s" % TOOLS_PATH

class Inference(EvaluatorMT):
    def __init__(self, trainer, data, params, parallel:bool):
        """
        Initialize evaluator.
        """
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder
        self.discriminator = trainer.discriminator
        self.data = data
        self.dico = data['dico']
        self.params = params
        self.parallel = parallel

        # create reference files for BLEU evaluation
        self.create_reference_files()
