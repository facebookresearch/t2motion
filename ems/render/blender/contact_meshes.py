import numpy as np

from .materials import body_material
from .foot_regions import leftFoot_vertices, rightFoot_vertices
# green
# GT_SMPL = body_material(0.009, 0.214, 0.029)
LEFT_SMPL = body_material(0.035, 0.415, 0.122)
RIGHT_SMPL = body_material(0.122, 0.414, 0.435)
DUO_SMPL = body_material(0.415, 0.1, 0.1)
# blue
# GEN_SMPL = body_material(0.022, 0.129, 0.439)
# Blues => cmap(0.87)
NON_SMPL = body_material(0.035, 0.322, 0.615)


class Meshes:
    def __init__(self, data, *, gt, mode, faces_path, canonicalize, always_on_floor, oldrender=True,**kwargs):
        data = prepare_meshes(data, canonicalize=canonicalize, always_on_floor=always_on_floor)

        self.faces = np.load(faces_path)
        self.data = data
        self.mode = mode
        self.oldrender = oldrender
        self.N = len(data)
        self.trajectory = data[:, :, [0, 1]].mean(1)
        self.contacts = calc_contacts(data)
        self.mat = []
        if len(self.contacts):
            for contact in self.contacts:
                if contact == 0:
                    self.mat.append(LEFT_SMPL)
                elif contact == 1:
                    self.mat.append(RIGHT_SMPL)
                elif contact == 2:
                    self.mat.append(DUO_SMPL)
                else:
                    self.mat.append(NON_SMPL)
        else:
            self.mat = [NON_SMPL]*self.N
        print(self.contacts)
        # raise RuntimeError("check contact type")
    def get_sequence_mat(self, frac):
        import matplotlib
        cmap = matplotlib.cm.get_cmap('Blues')
        # begin = 0.60
        # end = 0.90
        begin = 0.50
        end = 0.90
        rgbcolor = cmap(begin + (end-begin)*frac)
        mat = body_material(*rgbcolor, oldrender=self.oldrender)
        return mat

    def get_root(self, index):
        return self.data[index].mean(0)

    def get_mean_root(self):
        return self.data.mean((0, 1))

    def load_in_blender(self, index, mat):
        vertices = self.data[index]
        faces = self.faces
        name = f"{str(index).zfill(4)}"

        from .tools import load_numpy_vertices_into_blender
        load_numpy_vertices_into_blender(vertices, faces, name, mat)

        return name

    def __len__(self):
        return self.N


def prepare_meshes(data, canonicalize=True, always_on_floor=False):
    if canonicalize:
        print("No canonicalization for now")

    # fix axis
    data[..., 1] = - data[..., 1]
    data[..., 0] = - data[..., 0]

    # Remove the floor
    data[..., 2] -= data[..., 2].min()

    # Put all the body on the floor
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]

    return data


def calc_contacts(data,fthresh=0.015):
    num_frames = data.shape[0]
    contacts = []
    for idx in range(num_frames):
        z_vals = data[idx,:,2]
        # left 0, right 1, duo 2
        print(z_vals[rightFoot_vertices].shape)
        # raise RuntimeError("check z value size")
        if z_vals[leftFoot_vertices].min()<=fthresh and z_vals[rightFoot_vertices].min()<=fthresh:
            contacts.append(2)
        elif z_vals[leftFoot_vertices].min()<= fthresh:
            contacts.append(0)
        elif z_vals[rightFoot_vertices].min()<= fthresh:
            contacts.append(1)
        else:
            contacts.append(-1)
    return contacts
            
