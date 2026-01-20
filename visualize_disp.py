@torch.no_grad()
def visualize_displacement_field(model, slice_idx=None):
    """可视化形变场的某个切片"""
    import matplotlib.pyplot as plt
    
    # 获取位移场
    warped_coords = model.warp.warp()[0]
    displacement = (warped_coords - model.warp.pts).detach().cpu().numpy()
    # [D, H, W, 3]
    
    if slice_idx is None:
        slice_idx = displacement.shape[0] // 2
    
    # 提取中间切片
    disp_slice = displacement[slice_idx]  # [H, W, 3]
    
    # 计算位移幅度
    magnitude = np.sqrt((disp_slice ** 2).sum(axis=-1))
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # X 分量
    im0 = axes[0].imshow(disp_slice[..., 0], cmap='RdBu_r')
    axes[0].set_title('X Displacement')
    plt.colorbar(im0, ax=axes[0])
    
    # Y 分量
    im1 = axes[1].imshow(disp_slice[..., 1], cmap='RdBu_r')
    axes[1].set_title('Y Displacement')
    plt.colorbar(im1, ax=axes[1])
    
    # Z 分量
    im2 = axes[2].imshow(disp_slice[..., 2], cmap='RdBu_r')
    axes[2].set_title('Z Displacement')
    plt.colorbar(im2, ax=axes[2])
    
    # 幅度
    im3 = axes[3].imshow(magnitude, cmap='hot')
    axes[3].set_title('Displacement Magnitude')
    plt.colorbar(im3, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig('displacement_visualization.png', dpi=150)
    plt.show()
    
    print(f"位移统计 (体素):")
    print(f"  X: [{disp_slice[..., 0].min():.2f}, {disp_slice[..., 0].max():.2f}]")
    print(f"  Y: [{disp_slice[..., 1].min():.2f}, {disp_slice[..., 1].max():.2f}]")
    print(f"  Z: [{disp_slice[..., 2].min():.2f}, {disp_slice[..., 2].max():.2f}]")
    print(f"  幅度: [0, {magnitude.max():.2f}]")

# 使用
visualize_displacement_field(model, slice_idx=100)