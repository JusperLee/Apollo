###
# Author: Kai Li
# Date: 2024-01-22 01:16:22
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2024-01-24 00:05:10
###
import torch
import torchaudio
import json
from typing import Any, Dict, List, Optional, Tuple
import os
from omegaconf import OmegaConf
import argparse
import pytorch_lightning as pl
import hydra
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
# from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig
import look2hear.system
import look2hear.datas
import look2hear.losses
import look2hear.models
from look2hear.metrics import MetricsTracker
from look2hear.utils import RankedLogger, instantiate, print_only
import warnings
warnings.filterwarnings("ignore")

class Owndata():
    def __init__(self, root):
        self.root = root
        self.data_lists = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file == "codec_wav.wav":
                    self.data_lists.append(os.path.join(root, file))
        
    def __len__(self):
        return len(self.data_lists)
    
    def __getitem__(self, idx):
        ori = self.data_lists[idx].replace("codec_wav.wav", "ori_wav.wav")
        ori_audio = torchaudio.load(ori)[0]
        codec_audio = torchaudio.load(self.data_lists[idx])[0]
        return ori_audio, codec_audio, ori

def test(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data_val = Owndata("./codec-test")
    
    cfg.model.pop("_target_", None)
    model = look2hear.models.GullFullband.from_pretrain(os.path.join(cfg.exp.dir, cfg.exp.name, "best_model.pth"), **cfg.model).cuda()
    model.cuda()
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name, "results/"), exist_ok=True)
    metrics = MetricsTracker(save_file=os.path.join(cfg.exp.dir, cfg.exp.name, "results/")+"metrics.csv")
    length = len(data_val)
    for idx in range(len(data_val)):
        ori_wav, codec_wav, key = data_val[idx]
        
        ori_wav = ori_wav.cuda()
        codec_wav = codec_wav.unsqueeze(0).cuda()
        with torch.no_grad():
            ests = model(codec_wav)
            torchaudio.save(key.replace("ori_wav.wav", "ests_wav.wav"), ests.squeeze(0).cpu(), 44100)
            metrics(ori_wav, ests, key)
            
        if idx % 10 == 0:
            dicts = metrics.update()
            print(f"Processed {idx}/{length} samples: SDR: {dicts['sdr']}, SI-SNR: {dicts['si-snr']}, VISQOL: {dicts['visqol']}")
    
    metrics.final()       
    print("Finished testing")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="Exps/Apollo/config.yaml",
        help="Full path to save best validation model",
    )
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.conf_dir)
    
    os.makedirs(os.path.join(cfg.exp.dir, cfg.exp.name), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.exp.dir, cfg.exp.name, "config.yaml"))
    
    test(cfg)
    