import torch.nn.functional as F
from advertorch.utils import to_one_hot
from advertorch.utils import calc_l2distsq
import numpy as np
import advertorch
import torch
L2DIST_UPPER = 1e10
INVALID_LABEL = -1
class STA(advertorch.attacks.SpatialTransformAttack):
    # from https://github.com/BorealisAI/advertorch/blob/master/advertorch/attacks/spatial.py
    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)
        batch_size = len(x)
        loss_coeffs = x.new_ones(batch_size) * self.initial_const
        final_l2dists = [L2DIST_UPPER] * batch_size
        final_labels = [INVALID_LABEL] * batch_size
        final_step = [INVALID_LABEL] * batch_size
        final_advs = x.clone() #fixing bug from advertorch

        # TODO: refactor the theta generation
        theta = torch.tensor([[[1., 0., 0.],
                               [0., 1., 0.]]]).to(x.device)
        theta = theta.repeat((x.shape[0], 1, 1))


        grid = F.affine_grid(theta, x.size())

        grid_ori = grid.clone()
        y_onehot = to_one_hot(y, self.num_classes).float()

        clip_min = np.ones(grid_ori.shape[:]) * -1
        clip_max = np.ones(grid_ori.shape[:]) * 1
        clip_bound = list(zip(clip_min.flatten(), clip_max.flatten()))
        grid_ret = grid.clone().data.cpu().numpy().flatten().astype(float)
        from scipy.optimize import fmin_l_bfgs_b
        for outer_step in range(self.search_steps):
            grid_ret, f, d = fmin_l_bfgs_b(
                self._loss_fn_spatial,
                grid_ret,
                args=(
                    x.clone().detach(),
                    y_onehot, loss_coeffs,
                    grid_ori.clone().detach()),
                maxiter=self.max_iterations,
                bounds=clip_bound,
                iprint=0,
                maxls=100,
            )
            grid = torch.from_numpy(
                grid_ret.reshape(grid_ori.shape)).float().to(x.device)
            adv_x = F.grid_sample(x.clone(), grid)
            l2s = calc_l2distsq(grid.data, grid_ori.data)
            output = self.predict(adv_x)
            self._update_if_better(
                adv_x.data, y, output.data, l2s, batch_size,
                final_l2dists, final_labels, final_advs,
                outer_step, final_step)

        return final_advs