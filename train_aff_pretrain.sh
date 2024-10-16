# deepspeed --master_port=24999 --include localhost:0 train_aff_revise.py 

deepspeed --master_port=24999 --include localhost:2 train_aff_pretrain.py \
    --stage="pretrain" \
    --data_path="ex) /home/ubuntu/LLava_pre/blip_laion_cc_sbu_558k.json" \
    --image_folder="ex) /home/ubuntu/LLava_pre" \
    --conv_version="plain" \
    --batch_size=128 \
    --epochs=10 \
    --lr=1e-3 \
    --train_mask_decoder=False \
    --exp_name="aff_pretrain_try_1" \
    --load_mm_projector_file_path="./" \
    --load_mm_projector=False 2>&1 | tee -a pretrain_print.txt