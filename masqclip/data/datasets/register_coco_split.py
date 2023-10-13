import os

from detectron2.data import DatasetCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

COCO_CATEGORIES_BASE48 = [
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1, 'name': 'person'}, 
    {'color': [119, 11, 32], 'isthing': 1, 'id': 2, 'name': 'bicycle'}, 
    {'color': [0, 0, 142], 'isthing': 1, 'id': 3, 'name': 'car'}, 
    {'color': [0, 0, 230], 'isthing': 1, 'id': 4, 'name': 'motorcycle'}, 
    {'color': [0, 80, 100], 'isthing': 1, 'id': 7, 'name': 'train'}, 
    {'color': [0, 0, 70], 'isthing': 1, 'id': 8, 'name': 'truck'}, 
    {'color': [0, 0, 192], 'isthing': 1, 'id': 9, 'name': 'boat'}, 
    {'color': [250, 0, 30], 'isthing': 1, 'id': 15, 'name': 'bench'}, 
    {'color': [165, 42, 42], 'isthing': 1, 'id': 16, 'name': 'bird'}, 
    {'color': [182, 182, 255], 'isthing': 1, 'id': 19, 'name': 'horse'}, 
    {'color': [0, 82, 0], 'isthing': 1, 'id': 20, 'name': 'sheep'}, 
    {'color': [174, 57, 255], 'isthing': 1, 'id': 23, 'name': 'bear'}, 
    {'color': [199, 100, 0], 'isthing': 1, 'id': 24, 'name': 'zebra'}, 
    {'color': [72, 0, 118], 'isthing': 1, 'id': 25, 'name': 'giraffe'}, 
    {'color': [255, 179, 240], 'isthing': 1, 'id': 27, 'name': 'backpack'}, 
    {'color': [209, 0, 151], 'isthing': 1, 'id': 31, 'name': 'handbag'}, 
    {'color': [0, 220, 176], 'isthing': 1, 'id': 33, 'name': 'suitcase'}, 
    {'color': [255, 99, 164], 'isthing': 1, 'id': 34, 'name': 'frisbee'}, 
    {'color': [92, 0, 73], 'isthing': 1, 'id': 35, 'name': 'skis'}, 
    {'color': [0, 228, 0], 'isthing': 1, 'id': 38, 'name': 'kite'}, 
    {'color': [145, 148, 174], 'isthing': 1, 'id': 42, 'name': 'surfboard'}, 
    {'color': [197, 226, 255], 'isthing': 1, 'id': 44, 'name': 'bottle'}, 
    {'color': [207, 138, 255], 'isthing': 1, 'id': 48, 'name': 'fork'}, 
    {'color': [9, 80, 61], 'isthing': 1, 'id': 50, 'name': 'spoon'}, 
    {'color': [84, 105, 51], 'isthing': 1, 'id': 51, 'name': 'bowl'}, 
    {'color': [74, 65, 105], 'isthing': 1, 'id': 52, 'name': 'banana'}, 
    {'color': [166, 196, 102], 'isthing': 1, 'id': 53, 'name': 'apple'}, 
    {'color': [208, 195, 210], 'isthing': 1, 'id': 54, 'name': 'sandwich'}, 
    {'color': [255, 109, 65], 'isthing': 1, 'id': 55, 'name': 'orange'}, 
    {'color': [0, 143, 149], 'isthing': 1, 'id': 56, 'name': 'broccoli'}, 
    {'color': [179, 0, 194], 'isthing': 1, 'id': 57, 'name': 'carrot'}, 
    {'color': [5, 121, 0], 'isthing': 1, 'id': 59, 'name': 'pizza'}, 
    {'color': [227, 255, 205], 'isthing': 1, 'id': 60, 'name': 'donut'}, 
    {'color': [153, 69, 1], 'isthing': 1, 'id': 62, 'name': 'chair'}, 
    {'color': [119, 0, 170], 'isthing': 1, 'id': 65, 'name': 'bed'}, 
    {'color': [0, 165, 120], 'isthing': 1, 'id': 70, 'name': 'toilet'}, 
    {'color': [183, 130, 88], 'isthing': 1, 'id': 72, 'name': 'tv'}, 
    {'color': [95, 32, 0], 'isthing': 1, 'id': 73, 'name': 'laptop'}, 
    {'color': [130, 114, 135], 'isthing': 1, 'id': 74, 'name': 'mouse'}, 
    {'color': [110, 129, 133], 'isthing': 1, 'id': 75, 'name': 'remote'}, 
    {'color': [79, 210, 114], 'isthing': 1, 'id': 78, 'name': 'microwave'}, 
    {'color': [178, 90, 62], 'isthing': 1, 'id': 79, 'name': 'oven'},
    {'color': [65, 70, 15], 'isthing': 1, 'id': 80, 'name': 'toaster'}, 
    {'color': [59, 105, 106], 'isthing': 1, 'id': 82, 'name': 'refrigerator'},
    {'color': [142, 108, 45], 'isthing': 1, 'id': 84, 'name': 'book'}, 
    {'color': [196, 172, 0], 'isthing': 1, 'id': 85, 'name': 'clock'}, 
    {'color': [95, 54, 80], 'isthing': 1, 'id': 86, 'name': 'vase'}, 
    {'color': [191, 162, 208], 'isthing': 1, 'id': 90, 'name': 'toothbrush'}
]

COCO_CATEGORIES_NOVEL17 = [
    {'color': [106, 0, 228], 'isthing': 1, 'id': 5, 'name': 'airplane'}, 
    {'color': [0, 60, 100], 'isthing': 1, 'id': 6, 'name': 'bus'}, 
    {'color': [255, 77, 255], 'isthing': 1, 'id': 17, 'name': 'cat'}, 
    {'color': [0, 226, 252], 'isthing': 1, 'id': 18, 'name': 'dog'}, 
    {'color': [120, 166, 157], 'isthing': 1, 'id': 21, 'name': 'cow'}, 
    {'color': [110, 76, 0], 'isthing': 1, 'id': 22, 'name': 'elephant'}, 
    {'color': [0, 125, 92], 'isthing': 1, 'id': 28, 'name': 'umbrella'}, 
    {'color': [188, 208, 182], 'isthing': 1, 'id': 32, 'name': 'tie'}, 
    {'color': [133, 129, 255], 'isthing': 1, 'id': 36, 'name': 'snowboard'}, 
    {'color': [134, 134, 103], 'isthing': 1, 'id': 41, 'name': 'skateboard'}, 
    {'color': [109, 63, 54], 'isthing': 1, 'id': 47, 'name': 'cup'}, 
    {'color': [151, 0, 95], 'isthing': 1, 'id': 49, 'name': 'knife'}, 
    {'color': [147, 186, 208], 'isthing': 1, 'id': 61, 'name': 'cake'}, 
    {'color': [3, 95, 161], 'isthing': 1, 'id': 63, 'name': 'couch'}, 
    {'color': [166, 74, 118], 'isthing': 1, 'id': 76, 'name': 'keyboard'}, 
    {'color': [127, 167, 115], 'isthing': 1, 'id': 81, 'name': 'sink'}, 
    {'color': [128, 76, 255], 'isthing': 1, 'id': 87, 'name': 'scissors'}
]

COCO_CATEGORIES_INSTANCES65 = [
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1, 'name': 'person'}, 
    {'color': [119, 11, 32], 'isthing': 1, 'id': 2, 'name': 'bicycle'}, 
    {'color': [0, 0, 142], 'isthing': 1, 'id': 3, 'name': 'car'}, 
    {'color': [0, 0, 230], 'isthing': 1, 'id': 4, 'name': 'motorcycle'}, 
    {'color': [106, 0, 228], 'isthing': 1, 'id': 5, 'name': 'airplane'}, 
    {'color': [0, 60, 100], 'isthing': 1, 'id': 6, 'name': 'bus'}, 
    {'color': [0, 80, 100], 'isthing': 1, 'id': 7, 'name': 'train'}, 
    {'color': [0, 0, 70], 'isthing': 1, 'id': 8, 'name': 'truck'}, 
    {'color': [0, 0, 192], 'isthing': 1, 'id': 9, 'name': 'boat'}, 
    {'color': [250, 0, 30], 'isthing': 1, 'id': 15, 'name': 'bench'}, 
    {'color': [165, 42, 42], 'isthing': 1, 'id': 16, 'name': 'bird'}, 
    {'color': [255, 77, 255], 'isthing': 1, 'id': 17, 'name': 'cat'}, 
    {'color': [0, 226, 252], 'isthing': 1, 'id': 18, 'name': 'dog'}, 
    {'color': [182, 182, 255], 'isthing': 1, 'id': 19, 'name': 'horse'}, 
    {'color': [0, 82, 0], 'isthing': 1, 'id': 20, 'name': 'sheep'}, 
    {'color': [120, 166, 157], 'isthing': 1, 'id': 21, 'name': 'cow'}, 
    {'color': [110, 76, 0], 'isthing': 1, 'id': 22, 'name': 'elephant'}, 
    {'color': [174, 57, 255], 'isthing': 1, 'id': 23, 'name': 'bear'}, 
    {'color': [199, 100, 0], 'isthing': 1, 'id': 24, 'name': 'zebra'}, 
    {'color': [72, 0, 118], 'isthing': 1, 'id': 25, 'name': 'giraffe'}, 
    {'color': [255, 179, 240], 'isthing': 1, 'id': 27, 'name': 'backpack'}, 
    {'color': [0, 125, 92], 'isthing': 1, 'id': 28, 'name': 'umbrella'}, 
    {'color': [209, 0, 151], 'isthing': 1, 'id': 31, 'name': 'handbag'}, 
    {'color': [188, 208, 182], 'isthing': 1, 'id': 32, 'name': 'tie'}, 
    {'color': [0, 220, 176], 'isthing': 1, 'id': 33, 'name': 'suitcase'}, 
    {'color': [255, 99, 164], 'isthing': 1, 'id': 34, 'name': 'frisbee'}, 
    {'color': [92, 0, 73], 'isthing': 1, 'id': 35, 'name': 'skis'}, 
    {'color': [133, 129, 255], 'isthing': 1, 'id': 36, 'name': 'snowboard'}, 
    {'color': [0, 228, 0], 'isthing': 1, 'id': 38, 'name': 'kite'}, 
    {'color': [134, 134, 103], 'isthing': 1, 'id': 41, 'name': 'skateboard'}, 
    {'color': [145, 148, 174], 'isthing': 1, 'id': 42, 'name': 'surfboard'}, 
    {'color': [197, 226, 255], 'isthing': 1, 'id': 44, 'name': 'bottle'}, 
    {'color': [109, 63, 54], 'isthing': 1, 'id': 47, 'name': 'cup'}, 
    {'color': [207, 138, 255], 'isthing': 1, 'id': 48, 'name': 'fork'}, 
    {'color': [151, 0, 95], 'isthing': 1, 'id': 49, 'name': 'knife'}, 
    {'color': [9, 80, 61], 'isthing': 1, 'id': 50, 'name': 'spoon'}, 
    {'color': [84, 105, 51], 'isthing': 1, 'id': 51, 'name': 'bowl'}, 
    {'color': [74, 65, 105], 'isthing': 1, 'id': 52, 'name': 'banana'}, 
    {'color': [166, 196, 102], 'isthing': 1, 'id': 53, 'name': 'apple'}, 
    {'color': [208, 195, 210], 'isthing': 1, 'id': 54, 'name': 'sandwich'}, 
    {'color': [255, 109, 65], 'isthing': 1, 'id': 55, 'name': 'orange'}, 
    {'color': [0, 143, 149], 'isthing': 1, 'id': 56, 'name': 'broccoli'}, 
    {'color': [179, 0, 194], 'isthing': 1, 'id': 57, 'name': 'carrot'}, 
    {'color': [5, 121, 0], 'isthing': 1, 'id': 59, 'name': 'pizza'}, 
    {'color': [227, 255, 205], 'isthing': 1, 'id': 60, 'name': 'donut'}, 
    {'color': [147, 186, 208], 'isthing': 1, 'id': 61, 'name': 'cake'}, 
    {'color': [153, 69, 1], 'isthing': 1, 'id': 62, 'name': 'chair'}, 
    {'color': [3, 95, 161], 'isthing': 1, 'id': 63, 'name': 'couch'}, 
    {'color': [119, 0, 170], 'isthing': 1, 'id': 65, 'name': 'bed'}, 
    {'color': [0, 165, 120], 'isthing': 1, 'id': 70, 'name': 'toilet'}, 
    {'color': [183, 130, 88], 'isthing': 1, 'id': 72, 'name': 'tv'}, 
    {'color': [95, 32, 0], 'isthing': 1, 'id': 73, 'name': 'laptop'}, 
    {'color': [130, 114, 135], 'isthing': 1, 'id': 74, 'name': 'mouse'},
    {'color': [110, 129, 133], 'isthing': 1, 'id': 75, 'name': 'remote'}, 
    {'color': [166, 74, 118], 'isthing': 1, 'id': 76, 'name': 'keyboard'}, 
    {'color': [79, 210, 114], 'isthing': 1, 'id': 78, 'name': 'microwave'}, 
    {'color': [178, 90, 62], 'isthing': 1, 'id': 79, 'name': 'oven'}, 
    {'color': [65, 70, 15], 'isthing': 1, 'id': 80, 'name': 'toaster'}, 
    {'color': [127, 167, 115], 'isthing': 1, 'id': 81, 'name': 'sink'}, 
    {'color': [59, 105, 106], 'isthing': 1, 'id': 82, 'name': 'refrigerator'}, 
    {'color': [142, 108, 45], 'isthing': 1, 'id': 84, 'name': 'book'}, 
    {'color': [196, 172, 0], 'isthing': 1, 'id': 85, 'name': 'clock'}, 
    {'color': [95, 54, 80], 'isthing': 1, 'id': 86, 'name': 'vase'}, 
    {'color': [128, 76, 255], 'isthing': 1, 'id': 87, 'name': 'scissors'}, 
    {'color': [191, 162, 208], 'isthing': 1, 'id': 90, 'name': 'toothbrush'}
]

assert len(COCO_CATEGORIES_BASE48) == 48
assert len(COCO_CATEGORIES_NOVEL17) == 17
assert len(COCO_CATEGORIES_INSTANCES65) == 65


_PREDEFINED_SPLITS = {
    # point annotations without masks
    "coco_instance_train_instances65": (
        "coco/train2017",
        "coco/annotations/ov_split_48_17/instances_train2017_instances65.json",
    ),
    "coco_instance_train_base48": (
        "coco/train2017",
        "coco/annotations/ov_split_48_17/instances_train2017_seen48.json",
    ),
    "coco_instance_train_novel17": (
        "coco/train2017",
        "coco/annotations/ov_split_48_17/instances_train2017_unseen17.json",
    ),
    "coco_instance_val_instances65": (
        "coco/val2017",
        "coco/annotations/ov_split_48_17/instances_val2017_instances65.json",
    ),
    "coco_instance_val_base48": (
        "coco/val2017",
        "coco/annotations/ov_split_48_17/instances_val2017_seen48.json",
    ),
    "coco_instance_val_novel17": (
        "coco/val2017",
        "coco/annotations/ov_split_48_17/instances_val2017_unseen17.json",
    ),
}


def _get_coco_ovsplit_instances_meta(dataset_name):
    if "instances65" in dataset_name:
        COCO_CATEGORIES_SPLIT = COCO_CATEGORIES_INSTANCES65
    elif "base48" in dataset_name:
        COCO_CATEGORIES_SPLIT = COCO_CATEGORIES_BASE48
    elif "novel17" in dataset_name:
        COCO_CATEGORIES_SPLIT = COCO_CATEGORIES_NOVEL17

    thing_ids = [k["id"] for k in COCO_CATEGORIES_SPLIT]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES_SPLIT]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_all_coco_ovsplit_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_coco_ovsplit_instances_meta(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco_ovsplit_instance(_root)
