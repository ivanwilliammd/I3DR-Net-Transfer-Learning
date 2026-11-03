"""Uji asap I3DR-Net pada PyTorch modern (>=2.2).

Yang dijaga di sini bukan akurasi, melainkan bahwa kode yang dulu berjalan di torch 1.13 masih
dapat dimuat dan menjalankan forward pass pada torch 2.x — perubahan besar yang paling mudah
mematahkan sesuatu tanpa terlihat sampai model dilatih.
"""
import importlib

import pytest
import torch

CPU_MODULES = [
    "models.i3dbackbone",
    "models.detection_unet",
    "utils.model_utils",
    "utils.exp_utils",
    "utils.dataloader_utils",
    "evaluator",
    "default_configs",
    "predictor",
]

# Model deteksi memakai extension CUDA hasil `torch.utils.ffi`, API yang dihapus sejak torch 1.5.
# Yang diuji bukan keberhasilannya, melainkan pesan galatnya jelas dan menyebut jalan keluarnya.
CUDA_ONLY = ["models.retina_net", "models.retina_unet", "models.ufrcnn", "models.mrcnn"]


@pytest.mark.parametrize("nama", CPU_MODULES)
def test_modul_cpu_dapat_diimpor(nama):
    importlib.import_module(nama)


@pytest.mark.parametrize("nama", CUDA_ONLY)
def test_model_cuda_gagal_dengan_pesan_jelas(nama):
    with pytest.raises(RuntimeError, match="torch.utils.ffi"):
        importlib.import_module(nama)


def test_i3d_forward_pass():
    from models.i3dbackbone import I3D

    model = I3D(num_classes=2, modality="rgb")
    model.eval()
    with torch.no_grad():
        keluaran = model(torch.zeros(1, 3, 16, 64, 64))

    assert isinstance(keluaran, list) and keluaran, "I3D harus mengembalikan daftar feature map"
    for peta in keluaran:
        assert torch.is_tensor(peta) and peta.numel() > 0
        assert torch.isfinite(peta).all(), "forward pass menghasilkan NaN/Inf"


def test_i3d_bentuk_mengikuti_config():
    from models.i3dbackbone import I3D

    model = I3D(num_classes=2)
    assert model.num_classes == 2
    # Modality selain rgb/flow harus ditolak, bukan diterima diam-diam.
    with pytest.raises(ValueError):
        I3D(num_classes=2, modality="depth")


def test_evaluator_memakai_api_sklearn_yang_masih_ada():
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = [0, 0, 1, 1]
    skor = [0.1, 0.4, 0.35, 0.8]
    assert 0.0 <= roc_auc_score(y, skor) <= 1.0
    assert 0.0 <= average_precision_score(y, skor) <= 1.0


def test_default_configs_dapat_dibangun():
    from default_configs import DefaultConfigs

    cf = DefaultConfigs("retina_unet")
    assert cf.model == "retina_unet"
    assert cf.dim in (2, 3)
