"""Extension CUDA lama — tidak dapat dimuat pada PyTorch modern.

Berkas ini dulu dihasilkan `torch.utils.ffi`, API yang **dihapus sejak PyTorch 1.5**. Repo ini
dipatok ke torch>=2.2 (CVE-2025-32434 kritis), jadi modul hasil generate itu tidak bisa diimpor.

Yang perlu dilakukan bila model deteksi (retina_net, retina_unet, ufrcnn, mrcnn) memang dipakai:
bangun ulang `cuda_functions/*/` memakai `torch.utils.cpp_extension` di mesin ber-GPU, lalu
kembalikan impor `_ext` di berkas ini.

Jalur yang berjalan tanpa CUDA: i3dbackbone, detection_unet, utils.*, evaluator, default_configs,
predictor.
"""

raise RuntimeError(__doc__.strip())
