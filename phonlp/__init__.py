'''
Author: WEY
Date: 2021-05-07 21:52:44
LastEditTime: 2021-05-08 16:51:09
'''
import os
import sys
os.chdir(sys.path[0])
sys.path.append('../../')
sys.path.append('../')
# -*- coding: utf-8 -*-
from phonlp.run_script import download, load

__version__ = "0.3.2"
__all__ = [
    "download",
    "load",
]
