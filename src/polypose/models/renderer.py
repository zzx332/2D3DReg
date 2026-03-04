import torch
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform
from jaxtyping import Float
from torchio import LabelMap, ScalarImage

from ..warp import NonRigid, PolyRigid, SE3Field
import numpy as np
import os

def save_drr_as_nifti(
    img_tensor,
    filepath,
    drr_obj,
    is_mask=False,
):
    """
    灏� DRR 鍥惧儚鎴� mask 淇濆瓨涓� NIfTI (.nii.gz) 鏍煎紡銆�
    
    Args:
        img_tensor: torch.Tensor [H, W] - 鍗曞紶鍥惧儚
        filepath: str - 淇濆瓨璺緞锛堝 'drr.nii.gz'锛�
        drr_obj: DRR 瀵硅薄 - 鐢ㄤ簬鎻愬彇鍑犱綍鍙傛暟
        is_mask: bool - 鏄惁涓� mask
    """
    import torch
    import numpy as np
    import os
    from torchio import ScalarImage, LabelMap
    
    # 杞崲涓� numpy
    img_np = img_tensor.detach().cpu().numpy()
    
    # NIfTI 闇€瑕佽嚦灏� 3D锛屾坊鍔犳繁搴︾淮搴� [D=1, H, W]
    if img_np.ndim == 2:
        img_np = img_np[np.newaxis, ...]  # [1, H, W]
    
    # 娣诲姞閫氶亾缁村害鐢ㄤ簬 torchio [C=1, D=1, H, W]
    img_np = img_np[..., np.newaxis]
    
    # 杞崲涓� torch tensor
    img_data = torch.from_numpy(img_np).float()
    
    # 鏋勫缓 affine 鐭╅樀锛堝儚绱� -> 鐗╃悊鍧愭爣 mm锛�
    # 浠� DRR 鎺㈡祴鍣ㄥ弬鏁版彁鍙�
    delx = drr_obj.detector.delx  # mm/pixel (鍒楁柟鍚�)
    dely = drr_obj.detector.dely  # mm/pixel (琛屾柟鍚�)
    x0 = drr_obj.detector.x0      # mm (鍒楄捣鐐�)
    y0 = drr_obj.detector.y0      # mm (琛岃捣鐐�)
    
    # 鏋勫缓 4x4 affine 鐭╅樀
    # 鏍煎紡: [[sx, 0, 0, ox],
    #        [0, sy, 0, oy],
    #        [0, 0, sz, oz],
    #        [0, 0, 0, 1]]
    affine = np.eye(4)
    affine[0, 0] = delx      # X 鏂瑰悜缂╂斁锛堝垪锛�
    affine[1, 1] = dely      # Y 鏂瑰悜缂╂斁锛堣锛�
    affine[2, 2] = 1.0       # Z 鏂瑰悜锛堟繁搴�=1锛屾棤缂╂斁锛�
    affine[0, 3] = x0        # X 鏂瑰悜鍋忕Щ
    affine[1, 3] = y0        # Y 鏂瑰悜鍋忕Щ
    affine[2, 3] = 0.0       # Z 鏂瑰悜鍋忕Щ
    
    # 鍒涘缓 torchio 鍥惧儚瀵硅薄
    if is_mask:
        img_obj = LabelMap(tensor=img_data.to(torch.uint8), affine=affine)
    else:
        img_obj = ScalarImage(tensor=img_data, affine=affine)
    
    # 淇濆瓨
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    img_obj.save(filepath)
    
    print(f"  鉁� Saved: {filepath} (shape: {img_np.shape}, spacing: [{delx:.3f}, {dely:.3f}, 1.0] mm)")
    
    return filepath

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

    def forward(self, pose: RigidTransform, subject_id, **kwargs) -> Float[torch.Tensor, "B 1 H W"]:
        """Render a DRR from the warped density and mask."""
        from torch.amp import autocast
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        # output_dir = rf'/data/zhouzhexin/ctdsa/extracted_data_shift_only/subject{subject_id:02d}'
        output_dir = rf'/data/zhouzhexin/ctdsa/extracted_data_shift_only1/subject{subject_id:02d}'
        # 浣跨敤 float16 杩涜璁＄畻
        with autocast(device_type='cuda', dtype=torch.float16):
            for i in range(11):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    if i > 0:
                        # self.warp.poses_rot.data = torch.nn.Parameter(torch.randn(3, 3) * 0.05).cuda()
                        self.warp.poses_xyz.data = torch.nn.Parameter(torch.randn(3, 3) * 5.0).cuda()
                    warped_density, warped_mask, displacement = self.warp()
                    self.save_warped_volumes(warped_density, warped_mask, displacement, output_dir=os.path.join(output_dir, 'warped_data'), prefix=f'warped{i}')
                    del displacement
                    # 鍙覆鏌揵atch绗竴涓猵ose
                    batch_size = len(pose)
                    imgs = []
                    masks = []
                    # if batch_size > 1:
                    for j in range(batch_size):
                        if j > 0:
                            continue
                        single_pose = RigidTransform(pose.matrix[j:j+1])
                        source, target = self.drr.detector(single_pose, calibration=None)
                        img = self.render(warped_density, None, source, target, **kwargs)
                        mask = self.render_first_label_vectorized(warped_mask, source, target)
                        imgs.append(img)
                        masks.append(mask)
                        # 娓呯悊涓存椂鍙橀噺
                        del source, target, img, mask
                     # 娓呯悊涓存椂鍙橀噺
                    img = torch.cat(imgs, dim=0)
                    mask = torch.cat(masks, dim=0)
                    img = img.float()
                    mask = mask.float()
                    img = self.drr.reshape_transform(img, batch_size=1)
                    mask = self.drr.reshape_transform(mask, batch_size=1)

                    img_path = os.path.join(os.path.join(output_dir, 'diffdrr'), f'drr{i}.nii.gz')
                    save_drr_as_nifti(
                        img[0, 0],
                        img_path,
                        self.drr,
                        is_mask=False
                    )
                    
                    # 淇濆瓨 Mask
                    mask_path = os.path.join(os.path.join(output_dir, 'diffdrr'), f'mask{i}.nii.gz')
                    save_drr_as_nifti(
                        mask[0, 0],
                        mask_path,
                        self.drr,
                        is_mask=True
                    )
                    del img, mask, warped_density, warped_mask

        print(d)
        return img, mask

    def save_warped_volumes(self, warped_density, warped_mask, displacement, output_dir=r'D:\dataset\CTA_DSA\DeepFluoro\extracted_data\17-1882\warped_data', prefix='warped'):
        """
        淇濆瓨鍙樺舰鍚庣殑瀵嗗害鍦哄拰鎺╃爜涓� nii.gz 鏍煎紡
        """
        import os
        import numpy as np
        import SimpleITK as sitk

        os.makedirs(output_dir, exist_ok=True)
        
        # 鍏抽敭淇敼锛歞etach() 鍒嗙姊害
        warped_density = warped_density.detach()
        warped_mask = warped_mask.detach()
        
        # 鑾峰彇浠垮皠鐭╅樀
        affine = self.drr.subject.volume.affine
        
        # 淇濆瓨瀵嗗害鍦� (浠� WHD 杞崲涓� DHW)
        # density_data = warped_density.permute(2, 1, 0)[None].cpu().float()
        density_data = warped_density[None].cpu().float()
        density_img = ScalarImage(tensor=density_data, affine=affine)
        density_path = os.path.join(output_dir, f'{prefix}_density.nii.gz')
        density_img.save(density_path)
        print(f"宸蹭繚瀛樺瘑搴﹀満鍒�: {density_path}")
        
        # 淇濆瓨鎺╃爜
        # mask_data = warped_mask.permute(2, 1, 0)[None].cpu().to(torch.uint8)
        mask_data = warped_mask[None].cpu().to(torch.uint8)
        mask_img = LabelMap(tensor=mask_data, affine=affine)
        mask_path = os.path.join(output_dir, f'{prefix}_mask.nii.gz')
        mask_img.save(mask_path)
        print(f"宸蹭繚瀛樻帺鐮佸埌: {mask_path}")
        displacement = displacement.detach().cpu().numpy()
        displacement_physical = np.einsum('dhwi,ij->dhwj', displacement, affine[:3, :3])
        
        # 灏� 3 涓垎閲忎綔涓虹嫭绔嬬殑 channel 淇濆瓨
        # [D, H, W, 3] -> [3, D, H, W]
        displacement_tensor = torch.from_numpy(displacement_physical).permute(3, 0, 1, 2)
        # 淇濆瓨涓哄閫氶亾鍥惧儚
        disp_img = ScalarImage(tensor=displacement_tensor, affine=affine)
        disp_path = os.path.join(output_dir, f'{prefix}_displacement.nii.gz')
        disp_img.save(disp_path)
        del density_data, mask_data, displacement_tensor
        print(f"宸蹭繚瀛樹綅绉诲満鍒�: {disp_path}")
        
    def render(self, density, mask, source, target, **kwargs):
        img = (target - source).norm(dim=-1).unsqueeze(1)
        source = self.drr.affine_inverse(source)
        target = self.drr.affine_inverse(target)
        img = self.drr.renderer(density, source, target, img, mask=mask, **kwargs)
        return img

    def render_first_label_vectorized(
        self,
        mask: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        max_samples: int = 128,
    ) -> torch.Tensor:
        """
        澶歭abel mask 鐨� first-hit锛堟渶鍓嶉潰闈�0鏍囩锛夋姇褰便€�
        mask: (D,H,W) or (B,D,H,W) or (B,1,D,H,W)  鍊间负 0..K
        source/target: (B,N,3)  锛堜竴鑸� target 鏄� detector 鍍忕礌鐐瑰湪涓栫晫鍧愭爣/浣撶礌鍧愭爣鐨勬槧灏勶級
        return: (B,N) 姣忔潯灏勭嚎鐨勭涓€涓潪0鏍囩锛堟壘涓嶅埌鍒�0锛�
        """
        import torch.nn.functional as F

        # -------- 0) 缁熶竴 mask 褰㈢姸鍒� (B,1,D,H,W) --------
        if mask.dim() == 3:
            D, H, W = mask.shape
            mask_b = mask[None, None]  # (1,1,D,H,W)
        elif mask.dim() == 4:
            # (B,D,H,W)
            Bm, D, H, W = mask.shape
            mask_b = mask[:, None]     # (B,1,D,H,W)
        elif mask.dim() == 5:
            # (B,1,D,H,W) 鎴� (B,C,D,H,W)锛涜繖閲屽彧鍙栫涓€閫氶亾
            Bm, Cm, D, H, W = mask.shape
            if Cm != 1:
                mask_b = mask[:, :1]
            else:
                mask_b = mask
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        # source/target: (B,N,3)
        B, N, _ = target.shape

        # 鑻� mask 鍙湁 batch=1锛屼絾 source/target 鏄� B>1锛屽氨 expand
        if mask_b.shape[0] == 1 and B > 1:
            mask_b = mask_b.expand(B, -1, -1, -1, -1)
        elif mask_b.shape[0] != B:
            raise ValueError(f"mask batch {mask_b.shape[0]} != target batch {B}")

        # -------- 1) 杞埌浣撶礌鍧愭爣锛堜綘宸叉湁锛�--------
        source_voxel = self.drr.affine_inverse(source)  # (B,N,3) 浣撶礌鍧愭爣绯�
        target_voxel = self.drr.affine_inverse(target)  # (B,N,3)

        # 灏勭嚎鏂瑰悜锛氫粠 target -> source
        ray_dir = source_voxel - target_voxel  # (B,N,3)

        # -------- 2) 閲囨牱鐐癸細points (B,N,S,3) --------
        t = torch.linspace(0, 1, max_samples, device=mask_b.device, dtype=mask_b.dtype)  # (S,)
        t = t.view(1, 1, max_samples, 1)  # (1,1,S,1)
        points = target_voxel.unsqueeze(2) + t * ray_dir.unsqueeze(2)  # (B,N,S,3)

        # -------- 3) 褰掍竴鍖栧埌 grid_sample 闇€瑕佺殑 [-1,1]锛屾敞鎰忛『搴� (x,y,z)=(W,H,D) --------
        # points[...,0] 鏄� D 杞�? 杩樻槸 x? 鍙栧喅浜庝綘 affine_inverse 鐨勫畾涔夛紒
        # 浣犲師浠ｇ爜鎸� (D,H,W) 鏉ュ綊涓€鍖� 0,1,2 涓変釜鍒嗛噺銆傝繖閲屼繚鐣欎綘鐨勫亣璁撅細
        # points[...,0] 瀵瑰簲 D锛宲oints[...,1] 瀵瑰簲 H锛宲oints[...,2] 瀵瑰簲 W銆�
        # 浣� grid_sample 鐨� grid 鏈€鍚庝竴缁撮『搴忔槸 (x,y,z) => (W,H,D)锛屾墍浠ヨ閲嶆帓锛�
        d = points[..., 0]
        h = points[..., 1]
        w = points[..., 2]

        d_norm = 2.0 * d / (D - 1) - 1.0
        h_norm = 2.0 * h / (H - 1) - 1.0
        w_norm = 2.0 * w / (W - 1) - 1.0

        # grid 鐨勬渶鍚庣淮蹇呴』鏄� (x,y,z) = (w_norm, h_norm, d_norm)
        grid = torch.stack([w_norm, h_norm, d_norm], dim=-1)  # (B,N,S,3)

        # reshape 鎴� (B, D_out=S, H_out=1, W_out=N, 3)
        grid = grid.permute(0, 2, 1, 3).contiguous().view(B, max_samples, 1, N, 3)

        # -------- 4) grid_sample锛氳緭鍑� (B,1,S,1,N) --------
        sampled = F.grid_sample(
            input=mask_b,           # (B,1,D,H,W)
            grid=grid,              # (B,S,1,N,3)
            mode="nearest",         # label 蹇呴』 nearest
            padding_mode="zeros",
            align_corners=True,
        )  # (B,1,S,1,N)

        sampled = sampled[:, 0, :, 0, :]  # (B,S,N)
        sampled = sampled.permute(0, 2, 1).contiguous()  # (B,N,S)

        # -------- 5) first-hit锛氭瘡鏉″皠绾垮彇绗竴涓潪0鏍囩 --------
        # sampled: (B,N,S), 鍊间负 0..K
        nonzero = sampled != 0  # bool (B,N,S)

        # 鎵惧埌 first index锛氱敤 argmax(绱) 鐨勬妧宸�
        # 鏂瑰紡锛氭妸闈為浂杞负0/1锛岀劧鍚庢壘绗竴涓�1鐨勪綅缃�
        # 鍏堝鐞嗏€滃叏涓�0鈥濈殑鎯呭喌
        idx = torch.arange(max_samples, device=sampled.device).view(1, 1, max_samples)  # (1,1,S)
        INF = max_samples + 1
        idx_masked = torch.where(nonzero, idx, torch.full_like(idx, INF))  # (B,N,S)
        first_idx = idx_masked.min(dim=2).values  # (B,N)

        # gather 瀵瑰簲鏍囩
        first_idx_clamped = first_idx.clamp(0, max_samples - 1).unsqueeze(2)  # (B,N,1)
        first_label = torch.gather(sampled, 2, first_idx_clamped).squeeze(2)  # (B,N)

        # 鍏ㄤ负0鐨勫皠绾跨疆0
        first_label = torch.where(first_idx < INF, first_label, torch.zeros_like(first_label))
        return first_label

    @torch.no_grad
    def warp_subject(self, affine=None, volume_dtype=torch.float32, mask_dtype=torch.uint8):
        """Warp the original volume (HU) and segmentation mask."""
        if affine is None:
            affine = self.drr.subject.volume.affine
        warped_volume, warped_mask = self.warp.warp_subject()
        warped_volume = ScalarImage(tensor=warped_volume[None].cpu().to(volume_dtype), affine=affine)
        warped_mask = LabelMap(tensor=warped_mask[None].cpu().to(mask_dtype), affine=affine)
        return warped_volume, warped_mask
