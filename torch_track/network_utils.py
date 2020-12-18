import torch
import torch.nn as nn
import numpy as np

# CoordConv class from https://github.com/walsvid/CoordConv/blob/master/coordconv.py
# I actually changed it almost completely because torch has a meshgrid function already, so no real point in not using it
class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        if self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            yy_range = torch.arange(dim_y) - dim_y/2.0
            xx_range = torch.arange(dim_x) - dim_x/2.0

            # This scales to -1 to 1
            yy_range /= (dim_y-1)/2.0
            xx_range /= (dim_x-1)/2.0
            yy_range += 1/(dim_y-1)
            xx_range += 1/(dim_y-1)

            grid_yy, grid_xx = torch.meshgrid(yy_range, xx_range)

            grid_yy = grid_yy.repeat(batch_size_shape, 1, 1, 1)
            grid_xx = grid_xx.repeat(batch_size_shape, 1, 1, 1)

            grid_yy = grid_yy.to(input_tensor.device)
            grid_xx = grid_xx.to(input_tensor.device)

            out = torch.cat([grid_yy, grid_xx, input_tensor], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(grid_yy, 2) + torch.pow(grid_xx, 2))
                out = torch.cat([rr, out], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            zz_range = torch.arange(dim_z) - dim_z/2.0
            yy_range = torch.arange(dim_y) - dim_y/2.0
            xx_range = torch.arange(dim_x) - dim_x/2.0

            # This scales to -1 to 1
            zz_range /= (dim_z-1)/2.0
            yy_range /= (dim_y-1)/2.0
            xx_range /= (dim_x-1)/2.0

            zz_range += 1/(dim_z-1)
            yy_range += 1/(dim_y-1)
            xx_range += 1/(dim_x-1)

            grid_zz, grid_yy, grid_xx = torch.meshgrid(zz_range, yy_range, xx_range)

            grid_zz = grid_zz.repeat(batch_size_shape, 1, 1, 1, 1)
            grid_yy = grid_yy.repeat(batch_size_shape, 1, 1, 1, 1)
            grid_xx = grid_xx.repeat(batch_size_shape, 1, 1, 1, 1)

            grid_zz = grid_zz.to(input_tensor.device)
            grid_yy = grid_yy.to(input_tensor.device)
            grid_xx = grid_xx.to(input_tensor.device)

            out = torch.cat([grid_zz, grid_yy, grid_xx, input_tensor], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(grid_zz, 2) + torch.pow(grid_yy, 2) + torch.pow(grid_xx, 2))
                out = torch.cat([rr, out], dim=1)
        else:
            raise NotImplementedError

        return out