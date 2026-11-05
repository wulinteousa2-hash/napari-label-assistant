"""Small real-device smoke check; not a GPU performance or determinism study."""
import json
from pathlib import Path
import numpy as np
import pytest


def test_cuda_forward_backward_and_inference_reload(tmp_path):
    import torch
    from napari_label_assistant_tools._quick_train import QuickTrainConfig,QuickTrainModel,build_unet,save_model,load_model
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(71)
    config=QuickTrainConfig(crop_pixels=32,input_pixels=32,base_channels=4,device='cuda')
    network=build_unet(4).to('cuda').train()
    x=torch.linspace(0,1,32*32).reshape(1,1,32,32).to('cuda')
    optimizer=torch.optim.AdamW(network.parameters(),lr=.001)
    optimizer.zero_grad();loss=network(x).square().mean();loss.backward();optimizer.step()
    assert torch.isfinite(loss)
    network.eval()
    with torch.no_grad():
        before=network(x).cpu().numpy(); repeated=network(x).cpu().numpy()
    model=QuickTrainModel(network,config,32,{},'cuda');path=tmp_path/'cuda.pt'
    save_model(model,path);loaded=load_model(path)
    with torch.no_grad(): cpu=loaded.network(x.cpu()).numpy()
    with torch.no_grad(): gpu_reload=loaded.network.to('cuda')(x).cpu().numpy()
    old_cudnn=torch.backends.cudnn.allow_tf32
    old_matmul=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_tf32=False
        with torch.no_grad(): strict_gpu=loaded.network(x).cpu().numpy()
    finally:
        torch.backends.cudnn.allow_tf32=old_cudnn
        torch.backends.cuda.matmul.allow_tf32=old_matmul
    (Path(__file__).resolve().parents[1]/'evidence'/'cuda_smoke.json').write_text(json.dumps(dict(
        device=torch.cuda.get_device_name(0),loss=float(loss.detach()),shape=list(before.shape),
        gpu_repeat_max_abs=float(np.max(np.abs(before-repeated))),gpu_cpu_reload_max_abs=float(np.max(np.abs(before-cpu))),
        same_gpu_reload_max_abs=float(np.max(np.abs(before-gpu_reload))),strict_gpu_cpu_max_abs=float(np.max(np.abs(strict_gpu-cpu))),
        default_cross_device_within_3e4_tolerance=bool(np.allclose(before,cpu,atol=.0003,rtol=.0003)),
        default_device_threshold_flips=int(np.count_nonzero((before>=0)!=(cpu>=0))),
        default_cudnn_allow_tf32=old_cudnn,default_matmul_allow_tf32=old_matmul,
        tolerance=dict(same_gpu_atol=.000001,same_gpu_rtol=.000001,strict_cross_device_atol=.0003,strict_cross_device_rtol=.0003),
        scope='one optimizer step, repeated and reloaded GPU forwards, CPU reload and TF32-disabled GPU inference; no long-run reproducibility claim'),indent=2))
    np.testing.assert_allclose(before,repeated,atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(before,gpu_reload,atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(strict_gpu,cpu,atol=3e-4,rtol=3e-4)
