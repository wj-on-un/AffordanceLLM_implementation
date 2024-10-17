# imp_affordancellm
This is an implementation of an arbitrary affordancellm. 

It is not perfect(Incomplete version).

Please let me know if there is anything that needs to be fixed.

The official affordancellm github page of the author is here. 

-> https://github.com/JasonQSY/AffordanceLLM

## Environment
```Shell
conda create -n [env] python=3.10
conda activate [env]
git clone https://github.com/haotian-liu/LLaVA
cd LLaVA
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Data download
AGD20K: https://github.com/Reagan1311/LOCATE

Depth images: I used dpt_beit_large. https://github.com/isl-org/MiDaS

## Run Code
1. pre-training
```Shell
bash train_aff_pretrain.sh
```
2. fine-tuning
```Shell
bash train_aff_finetune_val.sh
```

### Something to note
※ I have referenced SAM and LISA code.

(https://github.com/dvlab-research/LISA)

※ Overall Feature dimension

※ Focal Loss

※ Output_upscaling part of mask_decoder
