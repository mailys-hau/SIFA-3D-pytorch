import torch
import torch.nn as nn

from monai.inferers import sliding_window_inference


class EnsembleModel(nn.Module):
    """
        Given a list of models this class implements the ensembling of
        these models by staking them together and taking the maximum
        predictions along each detected class.
    """
    def __init__(self, models, outchannels, roi_size=(160, 160, 160), sw_batch_size=4):
        super().__init__()
        self.models = models
        self.outchannels = outchannels
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size

    def forward(self, x):
        softmax = torch.nn.Softmax(dim=1)
        out_e = torch.zeros([x.shape[0], len(self.models), self.outchannels, *x.shape[2:]])

        for m, model in enumerate(self.models):
            if m == 1:
                out = model(x)

                if isinstance(out, list):
                    background = [sum([out[i].narrow(1, 0, 1) for i in range(len(out))])]
                    foreground = [out[i].narrow(1, 1, 1) for i in range(len(out))]

                    out = torch.cat(background + foreground, dim=1)
            else:
                out = sliding_window_inference(x, self.roi_size, self.sw_batch_size, model)

            out_e[:, m] = softmax(out)

        return torch.mean(out_e, dim=1)
