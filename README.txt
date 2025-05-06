# Learning Enhancement From Degradation: A Diffusion Model For Fundus Image Enhancement
Due to hardware limitations we were unable to completely run the finetuning, however it does run.

To run:
Create and activate a virtual environment using python 3.9.x
downgrade pip to 23.3.2
pip install -r requirements.txt
Then from the root directory, run:
python -m LED.scripts.finetune
or
accelerate launch --mixed_precision fp16 --gpu_ids 0 --num_processes 1 LED/script/finetune.py


## Acknowledgement 
Thanks for [PCENet](https://github.com/HeverLaw/PCENet-Image-Enhancement), [ArcNet](https://github.com/liamheng/Annotation-free-Fundus-Image-Enhancement) and [SCRNet](https://github.com/liamheng/Annotation-free-Fundus-Image-Enhancement) for sharing their powerful pre-trained weights! Thansk for [diffusers](https://github.com/huggingface/diffusers) for sharing codes.

LED is from [LED](https://github.com/QtacierP/LED), thank you.

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
