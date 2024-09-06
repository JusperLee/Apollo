###
# Author: Kai Li
# Date: 2021-06-22 12:41:36
# LastEditors: Please set LastEditors
# LastEditTime: 2022-06-05 14:48:00
###
import csv
from sympy import im
import torch
import numpy as np
import logging
import os
import librosa
from torch_mir_eval.separation import bss_eval_sources
import fast_bss_eval
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2

logger = logging.getLogger(__name__)

def is_silent(wav, threshold=1e-4):
    return torch.sum(wav ** 2) / wav.numel() < threshold

class MetricsTracker:
    def __init__(self, save_file: str = ""):
        self.all_sdrs = []
        self.all_sisnrs = []
        self.all_visqols = []
        
        csv_columns = ["snt_id", "sdr", "si-snr", "visqol"]
        self.visqol_config = visqol_config_pb2.VisqolConfig()
        self.visqol_config.audio.sample_rate = 48000
        self.visqol_config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
        self.visqol_config.options.svr_model_path = os.path.join(os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
        self.visqol_api = visqol_lib_py.VisqolApi()
        self.visqol_api.Create(self.visqol_config)
        
        self.results_csv = open(save_file, "w")
        self.writer = csv.DictWriter(self.results_csv, fieldnames=csv_columns)
        self.writer.writeheader()
        
    def __call__(self, clean, estimate, key):
        sisnr = fast_bss_eval.si_sdr(clean.unsqueeze(0), estimate.unsqueeze(0), zero_mean=True).mean()
        sdr = fast_bss_eval.sdr(clean.unsqueeze(0), estimate.unsqueeze(0), zero_mean=True).mean()
        
        clean = librosa.resample(clean.squeeze(0).mean(0).cpu().numpy(), orig_sr=44100, target_sr=48000).astype(np.float64)
        estimate = librosa.resample(estimate.squeeze(0).mean(0).cpu().numpy(), orig_sr=44100, target_sr=48000).astype(np.float64)
        
        visqol = self.visqol_api.Measure(clean, estimate).moslqo
        # import pdb; pdb.set_trace()
        row = {
            "snt_id": key,
            "sdr": sdr.item(),
            "si-snr": sisnr.item(),
            "visqol": visqol
        }
        
        self.writer.writerow(row)
        # Metric Accumulation
        self.all_sdrs.append(sdr.item())
        self.all_sisnrs.append(sisnr.item())
        self.all_visqols.append(visqol)
    
    def update(self, ):
        return {"sdr": np.array(self.all_sdrs).mean(),
                "si-snr": np.array(self.all_sisnrs).mean(),
                "visqol": np.array(self.all_visqols).mean()}
        
    def final(self,):
        row = {
            "snt_id": "avg",
            "sdr": np.array(self.all_sdrs).mean(),
            "si-snr": np.array(self.all_sisnrs).mean(),
            "visqol": np.array(self.all_visqols).mean()
        }
        self.writer.writerow(row)
        row = {
            "snt_id": "std",
            "sdr": np.array(self.all_sdrs).std(),
            "si-snr": np.array(self.all_sisnrs).std(),
            "visqol": np.array(self.all_visqols).std()
        }
        self.writer.writerow(row)
        self.results_csv.close()
