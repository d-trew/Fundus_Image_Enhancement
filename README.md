# Learning Enhancement From Degradation: A Diffusion Model For Fundus Image Enhancement

## Acknowledgement 
Thanks for [PCENet](https://github.com/HeverLaw/PCENet-Image-Enhancement), [ArcNet](https://github.com/liamheng/Annotation-free-Fundus-Image-Enhancement) and [SCRNet](https://github.com/liamheng/Annotation-free-Fundus-Image-Enhancement) for sharing their powerful pre-trained weights! Thansk for [diffusers](https://github.com/huggingface/diffusers) for sharing codes.

LED is from [LED](https://github.com/QtacierP/LED), thank you.

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

```
com3025cw
├─ commands.txt
├─ CycleGAN for image degredation.ipynb
├─ LED
│  ├─ .ipynb_checkpoints
│  │  └─ example-checkpoint.ipynb
│  ├─ configs
│  │  └─ train_led.yaml
│  ├─ docs
│  │  ├─ Continuous.png
│  │  ├─ example.jpeg
│  │  ├─ jpeg_loss.png
│  │  ├─ led.gif
│  │  ├─ mf_loss.png
│  │  ├─ OC.png
│  │  ├─ performance.png
│  │  └─ vessels.png
│  ├─ example.ipynb
│  ├─ led
│  │  ├─ backends
│  │  │  ├─ arcnet
│  │  │  │  ├─ backend.py
│  │  │  │  ├─ network.py
│  │  │  │  └─ __init__.py
│  │  │  ├─ base_backend.py
│  │  │  ├─ isecret
│  │  │  │  ├─ backend.py
│  │  │  │  ├─ network.py
│  │  │  │  └─ __init__.py
│  │  │  ├─ pcenet
│  │  │  │  ├─ backend.py
│  │  │  │  ├─ network.py
│  │  │  │  └─ __init__.py
│  │  │  └─ scrnet
│  │  │     ├─ backend.py
│  │  │     ├─ network.py
│  │  │     └─ __init__.py
│  │  ├─ data
│  │  │  └─ __init__.py
│  │  ├─ models
│  │  │  ├─ default_config.yaml
│  │  │  ├─ ema.py
│  │  │  ├─ unet.py
│  │  │  └─ __init__.py
│  │  ├─ pipelines
│  │  │  ├─ led_pipeline.py
│  │  │  └─ __init__.py
│  │  ├─ trainers
│  │  │  ├─ fine-tuned_trainer.py
│  │  │  ├─ led_trainer.py
│  │  │  └─ utils.py
│  │  └─ __init__.py
│  ├─ outputs
│  │  └─ dummy-checkpoint
│  ├─ pretrained_weights
│  ├─ script
│  │  ├─ dummy_checkpoint.py
│  │  ├─ finetune.py
│  │  └─ train.py
│  └─ __init__.py
├─ LICENSE
├─ main.py
├─ README.md
└─ requirements.txt

```
```