import matplotlib.pyplot as plt
import seaborn as sns
import torch
from diffdrr.data import read, load_example_ct
from pathlib import Path
from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr
from xvr.renderer import initialize_drr
from diffdrr.pose import RigidTransform

sns.set_context("talk")
KWARGS = dict(
    labels=None,
    orientation="PA",
    height=1436,
    width=1436,
    sdd=1020.0,
    delx=0.194,
    dely=0.194,
    x0=0.0,
    y0=0.0,
    reverse_x_axis=True,
    # renderer="trilinear",
    renderer="siddon",
    drr_kwargs={
        "voxel_shift": 0.0,
        # "patch_size": 256
    },
    read_kwargs={"bone_attenuation_multiplier": 2.0},
)
# Read in the volume and get its origin and spacing in world coordinates
# subject = load_example_ct(bone_attenuation_multiplier=2.0)
data_path = Path(r"D:\dataset\CTA_DSA\DeepFluoro\xvr-data\deepfluoro\subject01")
volume = data_path / "volume.nii.gz"
mask = data_path / "mask.nii.gz"
pose, *_ = torch.load(data_path / "xrays" / "010.pt", weights_only=False)["pose"]
pose = RigidTransform(pose).cuda()
# subject = read(volume, mask)
# Initialize the DRR module for generating synthetic X-rays
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
drr = initialize_drr(volume, mask, **KWARGS).to(device)
drr.rescale_detector_(0.5)
# # Specify the C-arm pose with a rotation (yaw, pitch, roll) and orientation (x, y, z)
# rot = torch.tensor([[0.0, 0.0, 0.0]], device=device)
# xyz = torch.tensor([[0.0, 850.0, 0.0]], device=device)
img = drr(pose)
# img = drr(pose, parameterization="euler_angles", convention="ZXY")
plot_drr(img, ticks=False)
plt.show()