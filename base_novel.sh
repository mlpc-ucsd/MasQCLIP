work_dir="output/base_novel"

# Progressive Distillation
python train_net.py --num-gpus 8 \
    --config-file configs/base-novel/coco-instance/teacher_R50_100k_base48.yaml \
    OUTPUT_DIR "${work_dir}/teacher"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_net.py --num-gpus 4 \
    --config-file configs/base-novel/coco-instance/student_R50_30k_base48.yaml \
    OUTPUT_DIR "${work_dir}/student" \
    MODEL.WEIGHTS "${work_dir}/teacher/model_final.pth"

# MasQ-Tuning
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_net.py --num-gpus 4 \
    --config-file configs/base-novel/coco-instance/masqclip_R50_bs4_10k_base48.yaml \
    OUTPUT_DIR "${work_dir}/masq" \
    MODEL.WEIGHTS "${work_dir}/student/model_final.pth"

# evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_net.py --eval-only --num-gpus 4 \
    --config-file configs/base-novel/coco-instance/masqclip_R50_bs4_10k_instance65.yaml \
    OUTPUT_DIR "${work_dir}/generalized" \
    MODEL.WEIGHTS "${work_dir}/masq/model_final.pth"
