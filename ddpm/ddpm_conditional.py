import torch

from ddpm.ddpm import DDPM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPM_conditional(DDPM):

    def __init__(self, network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device) -> None:
        super().__init__(network, num_timesteps)

    def reverse(self, x, t, label):
        # The network return the estimation of the noise we added

        return self.network.forward(x, t, label)
