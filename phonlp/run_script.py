'''
Author: WEY
Date: 2021-05-07 21:52:44
LastEditTime: 2021-05-11 18:05:19
'''
# -*- coding: utf-8 -*-
import os
import sys
os.chdir(sys.path[0])
sys.path.append('../../')
sys.path.append('../')
import gdown
import torch
from phonlp.annotate_model import JointModel
from phonlp.models.common import utils as util
from phonlp.models.ner.vocab import MultiVocab
from transformers import AutoConfig, AutoTokenizer


def download(save_dir, url="https://public.vinai.io/phonlp.pt"):
    util.ensure_dir(save_dir)
    if save_dir[len(save_dir) - 1] == "/":
        model_file = save_dir + "phonlp.pt"
    else:
        model_file = save_dir + "/phonlp.pt"
    gdown.download(url, model_file)


def load(save_dir="./"):
    if save_dir[len(save_dir) - 1] == "/":
        model_file = save_dir + "phonlp.pt"
    else:
        model_file = save_dir + "/phonlp.pt"
    print("Loading model from: {}".format(model_file))
    checkpoint = torch.load(model_file, lambda storage, loc: storage)
    args = checkpoint["config"]
    vocab = MultiVocab.load_state_dict(checkpoint["vocab"])
    # load model
    tokenizer = AutoTokenizer.from_pretrained(args["pretrained_lm"], use_fast=False)
    config_phobert = AutoConfig.from_pretrained(args["pretrained_lm"], output_hidden_states=True)
    model = JointModel(args, vocab, config_phobert, tokenizer)
    model.load_state_dict(checkpoint["model"], strict=False)
    if torch.cuda.is_available() is False:
        model.to(torch.device("cpu"))
    else:
        model.to(torch.device("cuda"))
    model.eval()
    return model


if __name__ == "__main__":
    download("./")
    model = load("./")
    text = "Tôi tên là Thế_Linh ."
    output = model.annotate(text=text)
    model.print_out(output)



    '''

    python3 run_phonlp.py --mode train --save_dir ./phonlp_tmp 
    --pretrained_lm "vinai/phobert-base" --lr 1e-5 --batch_size 8  
    --num_epoch 40 --lambda_pos 0.4 --lambda_ner 0.2 --lambda_dep 0.4 
    --train_file_pos ../sample_data/pos_train.txt --eval_file_pos ../sample_data/pos_valid.txt 
    --train_file_ner ../sample_data/ner_train.txt --eval_file_ner ../sample_data/ner_valid.txt 
    --train_file_dep ../sample_data/dep_train.conll --eval_file_dep ../sample_data/dep_valid.conll
    
    '''
