import argparse
import numpy as np
import torch

from models import MeshXL, train_aug, MeshTokenizer
from dataset import MeshDataset

def duplicates_detection(data_dict: dict, 
                         n_discrete_size: int):
    args_dict = {
        "n_discrete_size": n_discrete_size
    }
    args = argparse.Namespace(**args_dict)
    tokenizer = MeshTokenizer(args)
    # model = MeshXL(args)
    # print(model)
    # aaa

    data_dict['vertices'], _, _ = train_aug(data_dict['vertices'].clone(), (data_dict['faces'].clone()))
    # data_dict['vertices'] = data_dict['vertices'].unsqueeze(0)
    data_dict['faces'] = data_dict['faces'].unsqueeze(0)
    tokenizer.tokenize(data_dict)

    tri_tokens = data_dict['input_ids'] \
                    .cpu() \
                    .numpy()[0, 1: -1] \
                    .reshape(-1, 3, 3)

    for idx, group in enumerate(tri_tokens):
        if np.array_equal(group[0], group[1]) or \
            np.array_equal(group[0], group[2]) or \
            np.array_equal(group[1], group[2]):
                # print(tri_tokens)
                return False
    
    return True