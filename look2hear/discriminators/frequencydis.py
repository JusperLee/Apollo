import torch
import torch.nn as nn
import numpy as np

class MultiFrequencyDiscriminator(nn.Module):
    def __init__(self, nch, window):
        super(MultiFrequencyDiscriminator, self).__init__()

        self.nch = nch
        self.window = window
        self.hidden_channels = 8
        self.eps = torch.finfo(torch.float32).eps
        self.discriminators = nn.ModuleList([FrequencyDiscriminator(2*nch, self.hidden_channels) for _ in range(len(self.window))])

    def forward(self, est, sample_rate=44100):

        B, nch, _ = est.shape
        assert nch == self.nch

        # normalize power
        est = est / (est.pow(2).sum((1,2)) + self.eps).sqrt().reshape(B, 1, 1)
        est = est.view(-1, est.shape[-1])

        est_outputs = []
        est_feature_maps = []

        for i in range(len(self.discriminators)):
            est_spec = torch.stft(est.float(), self.window[i], self.window[i]//2, 
                                  window=torch.hann_window(self.window[i]).to(est.device).float(), 
                                  return_complex=True)
            est_RI = torch.stack([est_spec.real, est_spec.imag], dim=1)
            est_RI = est_RI.view(B, nch*2, est_RI.shape[-2], est_RI.shape[-1]).type(est.type())

            valid_enc = int(est_RI.shape[2] * sample_rate / 44100)
            est_out, est_feat_map = self.discriminators[i](est_RI[:,:,:valid_enc].contiguous())
            est_outputs.append(est_out)
            est_feature_maps.append(est_feat_map)

        return est_outputs, est_feature_maps


class FrequencyDiscriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels=512):
        super(FrequencyDiscriminator, self).__init__()

        self.eps = torch.finfo(torch.float32).eps
        self.discriminator = nn.ModuleList()
        self.discriminator += [
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels*2, hidden_channels*4, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels*4, hidden_channels*8, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels*8, hidden_channels*16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(hidden_channels*16, hidden_channels*32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Conv2d(hidden_channels*32, 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        ]

    def forward(self, x):
        hiddens = []
        for layer in self.discriminator:
            x = layer(x)
            hiddens.append(x)
        return x, hiddens[:-1]