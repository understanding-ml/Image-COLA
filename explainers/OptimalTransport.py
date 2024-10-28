import ot
import torch

class WassersteinDivergence:
    def __init__(self, reg=1):
        self.nu = None
        self.reg = reg

    def distance(self, y_s: torch.tensor, y_t: torch.tensor, weights: torch.tensor, delta):
        if delta < 0 or delta > 0.5:
            raise ValueError("Delta should be between 0 and 0.5")

        y_s = y_s.squeeze()
        y_t = y_t.squeeze()

        y_s_filtered = y_s[weights > 0]
        y_t_filtered = y_t[weights > 0]
        proj_y_s_dist_mass = torch.ones(len(y_s_filtered)) / len(y_s_filtered)
        proj_y_t_dist_mass = torch.ones(len(y_t_filtered)) / len(y_t_filtered)

        trimmed_M_y = ot.dist(
            y_s_filtered.reshape(y_s_filtered.shape[0], 1),
            y_t_filtered.reshape(y_t_filtered.shape[0], 1),
            metric="sqeuclidean",
        ).to("cpu")

        trimmed_nu = ot.emd(proj_y_s_dist_mass, proj_y_t_dist_mass, trimmed_M_y)
        dist = torch.sum(trimmed_nu * trimmed_M_y) * (1 / (1 - 2 * delta))

        return dist