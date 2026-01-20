import torch
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform
from jaxtyping import Float
from torchio import LabelMap, ScalarImage

from ..warp import NonRigid, PolyRigid, SE3Field


class DeformableRenderer(torch.nn.Module):
    """
    Render a DRR from a volume warped with a displacement field.
    """

    def __init__(
        self,
        drr: DRR,  # DRR containing a subject (volume and mask) to warp
        warp: str,  # Type of warp to use (polyrigid, nonrigid, se3)
        **kwargs,  # Additional arguments for the warp
    ):
        super().__init__()
        self.drr = drr
        if warp == "polyrigid":
            self.warp = PolyRigid(drr, **kwargs)
        elif warp == "nonrigid":
            self.warp = NonRigid(drr, **kwargs)
        elif warp == "se3":
            self.warp = SE3Field(drr, **kwargs)
        else:
            raise ValueError(f"Invalid warp: {warp}")

    def forward(self, pose: RigidTransform, **kwargs) -> Float[torch.Tensor, "B 1 H W"]:
        """Render a DRR from the warped density and mask."""
        # warped_density, warped_mask, displacement = self.warp()
        # self.save_warped_volumes(warped_density, warped_mask, displacement, prefix=f'warped{i}')
        # del warped_density, warped_mask, displacement
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        with torch.no_grad():
            for i in range(10):
                # 添加随机扰动生成不同的变形
                if i > 0:  # 第一次使用原始参数
                    self.warp.poses_rot.data = torch.nn.Parameter(torch.randn(3, 3) * 0.15).cuda()
                    self.warp.poses_xyz.data = torch.nn.Parameter(torch.randn(3, 3) * 15.0).cuda()
                
                # 生成变形数据
                warped_density, warped_mask, displacement = self.warp()
                
                # 保存
                self.save_warped_volumes(
                    warped_density, warped_mask, displacement,
                    prefix=f'warped{i}'
                )
                
                # 清理
                del warped_density, warped_mask, displacement
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                print(f"✓ 变形 {i+1}/10 保存完成")
        source, target = self.drr.detector(pose, calibration=None)
        img = self.render(warped_density, warped_mask, source, target, **kwargs)
        img = self.drr.reshape_transform(img, batch_size=len(pose))
        return img

    def save_warped_volumes(self, warped_density, warped_mask, displacement, output_dir=r'D:\dataset\CTA_DSA\DeepFluoro\extracted_data\18-2800\warped_data', prefix='warped'):
        """
        保存变形后的密度场和掩码为 nii.gz 格式
        """
        import os
        import numpy as np
        import SimpleITK as sitk

        os.makedirs(output_dir, exist_ok=True)
        
        # 关键修改：detach() 分离梯度
        warped_density = warped_density.detach()
        warped_mask = warped_mask.detach()
        
        # 获取仿射矩阵
        affine = self.drr.subject.volume.affine
        
        # 保存密度场 (从 WHD 转换为 DHW)
        # density_data = warped_density.permute(2, 1, 0)[None].cpu().float()
        density_data = warped_density[None].cpu().float()
        density_img = ScalarImage(tensor=density_data, affine=affine)
        density_path = os.path.join(output_dir, f'{prefix}_density.nii.gz')
        density_img.save(density_path)
        print(f"已保存密度场到: {density_path}")
        
        # 保存掩码
        # mask_data = warped_mask.permute(2, 1, 0)[None].cpu().to(torch.uint8)
        mask_data = warped_mask[None].cpu().to(torch.uint8)
        mask_img = LabelMap(tensor=mask_data, affine=affine)
        mask_path = os.path.join(output_dir, f'{prefix}_mask.nii.gz')
        mask_img.save(mask_path)
        print(f"已保存掩码到: {mask_path}")
        displacement = displacement.detach().cpu().numpy()
        displacement_physical = np.einsum('dhwi,ij->dhwj', displacement, affine[:3, :3])
        
        # 将 3 个分量作为独立的 channel 保存
        # [D, H, W, 3] -> [3, D, H, W]
        displacement_tensor = torch.from_numpy(displacement_physical).permute(3, 0, 1, 2)
        # 保存为多通道图像
        disp_img = ScalarImage(tensor=displacement_tensor, affine=affine)
        disp_path = os.path.join(output_dir, f'{prefix}_displacement.nii.gz')
        disp_img.save(disp_path)
        del density_data, mask_data, displacement_tensor
        print(f"已保存位移场到: {disp_path}")
        
    def render(self, density, mask, source, target, **kwargs):
        img = (target - source).norm(dim=-1).unsqueeze(1)
        source = self.drr.affine_inverse(source)
        target = self.drr.affine_inverse(target)
        img = self.drr.renderer(density, source, target, img, mask=mask, **kwargs)
        return img

    @torch.no_grad
    def warp_subject(self, affine=None, volume_dtype=torch.float32, mask_dtype=torch.uint8):
        """Warp the original volume (HU) and segmentation mask."""
        if affine is None:
            affine = self.drr.subject.volume.affine
        warped_volume, warped_mask = self.warp.warp_subject()
        warped_volume = ScalarImage(tensor=warped_volume[None].cpu().to(volume_dtype), affine=affine)
        warped_mask = LabelMap(tensor=warped_mask[None].cpu().to(mask_dtype), affine=affine)
        return warped_volume, warped_mask
