import numpy as np
import torch 
import os
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

class NoiselessBackwardFK:
    def __init__(self, y, obs_idx, miss_idx, alpha_bars, sigmas, model, dx):
        self.y = y
        self.obs_idx = obs_idx
        self.miss_idx = miss_idx
        self.dy = len(obs_idx)
        self.alpha_bars = alpha_bars
        self.sigmas = sigmas
        self.model = model
        self.dx = dx
        self.nsteps = len(alpha_bars) - 1

    @torch.no_grad()
    def chi(self, x, t):
        x = np.atleast_2d(x)
        N = x.shape[0]
        img_size = int(np.sqrt(self.dx))
        x_img = x.reshape(N, 1, img_size, img_size)
        x_tensor = torch.tensor(x_img, dtype=torch.float32, device=device)
        t_tensor = torch.full((N,), t, device=device, dtype=torch.long)
        eps = self.model(x_tensor, t_tensor).cpu().numpy().reshape(N, -1)
        return x / np.sqrt(self.alpha_bars[t]) - np.sqrt(1 - self.alpha_bars[t]) * eps

    def m(self, x_next, t):
        chi = self.chi(x_next, t + 1)
        sigma2 = self.sigmas[t + 1]**2
        alpha_bar = self.alpha_bars[t + 1]
        Kt = sigma2 / (sigma2 + 1 - alpha_bar)
        mean = chi.copy()
        mean[:, self.obs_idx] = Kt * np.sqrt(alpha_bar) * self.y + (1 - Kt) * chi[:, self.obs_idx]
        return mean

    def M0(self, N):
        particles = np.zeros((N, self.dx))
        alpha_n = self.alpha_bars[-1]
        sigma_n = self.sigmas[-1]
        Kn = sigma_n**2 / (sigma_n**2 + 1 - alpha_n)
        mean_obs = Kn * np.sqrt(alpha_n) * self.y
        cov_obs = (1 - alpha_n) * Kn * np.eye(self.dy)
        particles[:, self.obs_idx] = np.random.multivariate_normal(mean_obs, cov_obs, size=N)
        particles[:, self.miss_idx] = np.random.randn(N, len(self.miss_idx))
        return particles

    def M(self, s, xp):
        mean = self.m(xp, s)
        alpha_s = self.alpha_bars[s]
        sigma_sp1 = self.sigmas[s + 1]
        Ks = sigma_sp1**2 / (sigma_sp1**2 + 1 - alpha_s)
        cov_obs = (1 - alpha_s) * Ks * np.eye(self.dy)
        new_obs = np.array([np.random.multivariate_normal(mean=m[self.obs_idx], cov=cov_obs) for m in mean])
        xp_new = xp.copy()
        xp_new[:, self.obs_idx] = new_obs
        return xp_new
    


class PseudoSMC:
    def __init__(self, fk, N=5, snapshot_dir :str |None = None):
        self.fk = fk
        self.N = N
        if snapshot_dir:
            self.snapshot_dir = snapshot_dir
            os.makedirs(snapshot_dir, exist_ok=True)

    def compute_weights(self, particles, s):
        mean = self.fk.m(particles, s)
        cov = self.fk.sigmas[s + 1]**2 + 1 - self.fk.alpha_bars[s]
        log_w = np.zeros(self.N)
        for i in range(self.N):
            diff = np.sqrt(self.fk.alpha_bars[s]) * self.fk.y - mean[i, self.fk.obs_idx]
            log_w[i] = -0.5 * diff @ diff / cov - 0.5 * self.fk.dy * np.log(2 * np.pi * cov)
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        if np.sum(w) == 0 or np.isnan(np.sum(w)):
            return np.ones(self.N) / self.N
        return w / np.sum(w)

    def resample(self, particles, weights):
        idx = np.random.choice(self.N, self.N, replace=True, p=weights)
        return particles[idx]

    def run(self, snapshot_every=5):
        particles = self.fk.M0(self.N)
        for s in reversed(range(self.fk.nsteps)):
            w = self.compute_weights(particles, s)
            particles = self.resample(particles, w)
            particles = self.fk.M(s, particles)
            if self.snapshot_dir and s % snapshot_every == 0:
                is_valid = np.all(np.isfinite(particles), axis=1)
                n_valid = np.sum(is_valid)
                n_degen = self.N - n_valid

                if n_valid > 0:
                    recon = particles[is_valid].mean(axis=0).reshape(28, 28)
                else:
                    recon = np.zeros((28,28))

                filename = os.path.join(self.snapshot_dir, f"step_{s:04d}.png")
                plt.imsave(filename, recon, cmap="gray")
                print(f"Step {s}: {n_valid} valid, {n_degen} degenerate particles")
                plt.imsave(os.path.join(self.snapshot_dir, f"step_{s:04d}.png"), recon, cmap="gray")
        return particles