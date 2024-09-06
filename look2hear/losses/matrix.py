import torch
from torch.nn.modules.loss import _Loss

class MultiSrcNegSDR(_Loss):
    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super().__init__()

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8

    def forward(self, ests, targets):
        if targets.size() != ests.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {ests.size()} instead"
            )
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_est = torch.mean(ests, dim=2, keepdim=True)
            targets = targets - mean_source
            ests = ests - mean_est
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src]
            pair_wise_dot = torch.sum(ests * targets, dim=2, keepdim=True)
            # [batch, n_src]
            s_target_energy = torch.sum(targets ** 2, dim=2, keepdim=True) + self.EPS
            # [batch, n_src, time]
            scaled_targets = pair_wise_dot * targets / s_target_energy
        else:
            # [batch, n_src, time]
            scaled_targets = targets
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = ests - targets
        else:
            e_noise = ests - scaled_targets
        # [batch, n_src]
        pair_wise_sdr = torch.sum(scaled_targets ** 2, dim=2) / (
            torch.sum(e_noise ** 2, dim=2) + self.EPS
        )
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)
        return -torch.mean(pair_wise_sdr, dim=-1).mean(0)