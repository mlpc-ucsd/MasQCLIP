work_dir="output/cross-dataset"

# Progressive Distillation
python train_net.py --num-gpus 8 \
    --config-file configs/cross-dataset/coco-train/panoptic-segmentation/teacher_R50_200k.yaml \
    OUTPUT_DIR "${work_dir}/train_coco/teacher"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_net.py --num-gpus 4 \
    --config-file configs/cross-dataset/coco-train/panoptic-segmentation/student_R50_30k.yaml \
    OUTPUT_DIR "${work_dir}/train_coco/student" \
    MODEL.WEIGHTS "${work_dir}/train_coco/teacher/model_final.pth"

# MasQ-Tuning
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_net.py --num-gpus 4 \
    --config-file configs/cross-dataset/coco-train/panoptic-segmentation/masqclip_R50_bs4_10k.yaml \
    OUTPUT_DIR "${work_dir}/train_coco/masq" \
    MODEL.WEIGHTS "${work_dir}/train_coco/student/model_final.pth"

# evaluation
model_path="${work_dir}/train_coco/masq/model_final.pth"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_net.py --eval-only --num-gpus 4 \
    --config-file configs/cross-dataset/test/ade20k-150/panoptic-segmentation/masqclip_R50_bs4_10k.yaml \
    OUTPUT_DIR "${work_dir}/test_ade20k_150" \
    MODEL.WEIGHTS $model_path

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_net.py --eval-only --num-gpus 4 \
    --config-file configs/cross-dataset/test/ade20k-full-847/semantic-segmentation/masqclip_R50_bs4_10k.yaml \
    OUTPUT_DIR "${work_dir}/test_ade20k_full" \
    MODEL.WEIGHTS $model_path

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_net.py --eval-only --num-gpus 4 \
    --config-file configs/cross-dataset/test/pascal-ctx-59/semantic-segmentation/masqclip_R50_bs4_10k.yaml \
    OUTPUT_DIR "${work_dir}/test_pascal_59" \
    MODEL.WEIGHTS $model_path

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_net.py --eval-only --num-gpus 4 \
    --config-file configs/cross-dataset/test/pascal-ctx-459/semantic-segmentation/masqclip_R50_bs4_10k.yaml \
    OUTPUT_DIR "${work_dir}/test_pascal_459" \
    MODEL.WEIGHTS $model_path
