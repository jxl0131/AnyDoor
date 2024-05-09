wget -t 0 -c -b 'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt'
wget -t 0 -c -b 'https://modelscope.cn/api/v1/models/iic/AnyDoor/repo?Revision=master&FilePath=dinov2_vitg14_pretrain.pth'
wget -c -b 'https://modelscope.cn/api/v1/models/iic/AnyDoor/repo?Revision=master&FilePath=epoch%3D1-step%3D8687.ckpt' -O epoch=1-step=8687.ckpt