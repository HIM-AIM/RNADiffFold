# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: yjcmydkzgj@gmail.com
"""

# import fm
# def choose_backbone(backbone_name):
#     if backbone_name == 'rna-fm':
#         backbone, backbone_alphabet = fm.pretrained.rna_fm_t12()
#     else:
#         raise Exception("Wrong Backbone Type! {}".format(backbone_name))
#
#     return backbone, backbone_alphabet

from ..pretrained import *

model_location = '/t9k/mnt/RNA-FM-main/redevelop/pretrained/RNA-FM_pretrained.pth'


def choose_backbone(backbone_name):
    if backbone_name == 'rna-fm':
        backbone, backbone_alphabet = rna_fm_t12(model_location=model_location)
    else:
        raise Exception("Wrong Backbone Type! {}".format(backbone_name))

    return backbone, backbone_alphabet
