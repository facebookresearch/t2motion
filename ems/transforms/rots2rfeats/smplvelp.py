from typing import Optional
import torch
from torch import Tensor
from einops import rearrange
import numpy as np
from ems.tools.easyconvert import matrix_to, nfeats_of, to_matrix
import ems.tools.geometry as geometry
from .base import Rots2Rfeats
from scipy.spatial.transform import Rotation

class SMPLVelP(Rots2Rfeats):
    def __init__(self, path: Optional[str] = None,
                 normalization: bool = False,
                 pose_rep: str = "rot6d",
                 canonicalize: bool = False,
                 offset: bool = True,
                 rotation: bool = False,
                 **kwargs) -> None:
        super().__init__(path=path, normalization=normalization)
        self.canonicalize = canonicalize
        self.pose_rep = pose_rep
        self.nfeats = nfeats_of(pose_rep)
        self.offset = offset
        self.rotation = rotation
        # self.targ_root = torch.Tensor([[ 0.0057, -0.0104,  0.9999],
        # [ 0.9994,  0.0351, -0.0053],
        # [-0.0350,  0.9993,  0.0106]])
        
        if self.rotation:
            z_rot = torch.from_numpy(Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32))
            x_rot = torch.from_numpy(Rotation.from_euler('x', 100, degrees=True).as_matrix().astype(np.float32))
            self.targ_root = torch.matmul(x_rot,z_rot)
        # if self.rotation:
        #     y_rot = torch.from_numpy(Rotation.from_euler('y', 70, degrees=True).as_matrix().astype(np.float32))
        #     x_rot = torch.from_numpy(Rotation.from_euler('x', 110, degrees=True).as_matrix().astype(np.float32))
        #     self.targ_root = torch.matmul(y_rot,x_rot)
    
    def forward(self, data) -> Tensor:
        matrix_poses, trans = data.rots, data.trans
        # matrix_poses: [nframes, 22, 3, 3]
        # trans: [nframes, 3]
        # extract the root gravity axis
        # for smpl it is the last coordinate
        
        if self.rotation:
            swap_trans = torch.zeros_like(trans)
            swap_trans[:,0] = trans[:,0]
            swap_trans[:,1] = trans[:,2]
            swap_trans[:,2] = trans[:,1]
            trans = swap_trans * -1.0
            trans[:,:2] = 0.0
        # trans = trans * 0.0
        
        
        # trans = torch.matmul(self.targ_root,trans.T).T
        # print(trans)
        # raise RuntimeError("check trans")
        root_y = trans[..., 2]
        trajectory = trans[..., [0, 1]]

        # Comoute the difference of trajectory (for X and Y axis)
        vel_trajectory = torch.diff(trajectory, dim=-2)
        # 0 for the first one => keep the dimentionality
        # print(vel_trajectory.size())
        vel_trajectory = torch.cat((0 * vel_trajectory[..., [0], :], vel_trajectory), dim=-2)

        # first normalize the data
        if self.canonicalize:
            nf = matrix_poses.size(0)
            global_orient = matrix_poses[..., 0, :, :]
            # rotate the root orientation to amass direction
            
            # left rotate
            if self.rotation:
                global_orient = torch.matmul(self.targ_root.expand(nf,3,3),global_orient)
            
            #right rotate
            # if self.rotation:
            #     reverse_orient = torch.zeros_like(global_orient)
            #     reverse_orient[0] = torch.matmul(global_orient[0],self.targ_root)
            #     for i in range(1, nf):
            #         reverse_orient[i] = torch.matmul(reverse_orient[0],torch.matmul(global_orient[0].inverse(),global_orient[i]))
            #     global_orient = reverse_orient
            # Remove the fist rotation along the vertical axis
            # construct this by extract only the vertical component of the rotation
            rot2d = geometry.matrix_to_axis_angle(global_orient[..., 0, :, :])
            rot2d[..., :2] = 0
            
            if self.offset:
                # add a bit more rotation
                rot2d[..., 2] += torch.pi/2

            rot2d = geometry.axis_angle_to_matrix(rot2d)

            # turn with the same amount all the rotations
            global_orient = torch.einsum("...kj,...kl->...jl", rot2d, global_orient)

            matrix_poses = torch.cat((global_orient[..., None, :, :],
                                      matrix_poses[..., 1:, :, :]), dim=-3)

            # Turn the trajectory as well
            vel_trajectory = torch.einsum("...kj,...lk->...lj", rot2d[..., :2, :2], vel_trajectory)

        poses = matrix_to(self.pose_rep, matrix_poses)
        features = torch.cat((root_y[..., None],
                              vel_trajectory,
                              rearrange(poses, "... joints rot -> ... (joints rot)")),
                             dim=-1)
        features = self.normalize(features)
        return features

    def extract(self, features):
        root_y = features[..., 0]
        vel_trajectory = features[..., 1:3]
        poses_features = features[..., 3:]
        poses = rearrange(poses_features,
                          "... (joints rot) -> ... joints rot", rot=self.nfeats)
        return root_y, vel_trajectory, poses

    def inverse(self, features):
        features = self.unnormalize(features)
        root_y, vel_trajectory, poses = self.extract(features)

        # integrate the trajectory
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        # First frame should be 0, but if infered it is better to ensure it
        trajectory = trajectory - trajectory[..., [0], :]

        # Get back the translation
        trans = torch.cat([trajectory, root_y[..., None]], dim=-1)
        matrix_poses = to_matrix(self.pose_rep, poses)

        from ems.transforms.smpl import RotTransDatastruct
        return RotTransDatastruct(rots=matrix_poses, trans=trans)
