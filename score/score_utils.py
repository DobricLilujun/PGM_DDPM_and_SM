import torch
import math


class AnnealedLangevinDynamic():
    def __init__(self, sigma_min, sigma_max, n_steps, annealed_step, score_fn, device, eps = 1e-1):

        # sigma_min: minimum sigmas of perturbation schedule 
        # sigma_max: maximum sigmas of perturbation schedule 
        # L: iteration step of Langevin dynamic
        # T: annelaed step of annealed Langevin dynamic
        # score_fn: trained score network
        # eps: coefficient of step size

        self.process = torch.exp(torch.linspace(start=math.log(sigma_max), end=math.log(sigma_min), steps = n_steps))
        self.step_size = eps * (self.process / self.process[-1] ) ** 2
        self.score_fn = score_fn
        self.annealed_step = annealed_step
        self.device = device
        
    # One iteration of annealed step
    def _one_annealed_step_iteration(self, x, idx):

        self.score_fn.eval()
        z, step_size = torch.randn_like(x).to(device = self.device), self.step_size[idx]
        x = x + 0.5 * step_size * self.score_fn(x, idx) + torch.sqrt(step_size) * z
        return x
        
    # One annealed step
    def _one_annealed_step(self, x, idx):

        for _ in range(self.annealed_step):
            x = self._one_annealed_step_iteration(x, idx)
        return x
        
    # One Langevin Step
    def _one_diffusion_step(self, x):

        for idx in range(len(self.process)):
            x = self._one_annealed_step(x, idx)
            yield x

    @torch.no_grad()
    def sampling(self, sampling_number, only_final=False):

        sample = torch.rand([sampling_number, 1, 28, 28]).to(device = self.device)
        sampling_list = []
        
        final = None
        for sample in self._one_diffusion_step(sample):
            final = sample
            if not only_final:
                sampling_list.append(final)
                

        return final if only_final else torch.stack(sampling_list)
