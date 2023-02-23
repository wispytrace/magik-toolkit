""" configurations for this project

author baiyu
"""
import os
from datetime import datetime


CHECKPOINT_PATH = 'checkpoint'

EPOCH = 200
MILESTONES = [40, 80, 120, 150]

#initial learning rate
#INIT_LR = 0.1

TIME_NOW = datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss')
LOG_DIR = 'runs'
SAVE_EPOCH = 10








