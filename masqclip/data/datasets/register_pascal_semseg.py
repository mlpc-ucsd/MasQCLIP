# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog

PASCAL_59_CATEGORIES = [
    {"category_id": 2, "name": "aeroplane"},  # 0
    {"category_id": 9, "name": "bag"},
    {"category_id": 18, "name": "bed"},
    {"category_id": 19, "name": "bedclothes"},
    {"category_id": 22, "name": "bench"},
    {"category_id": 23, "name": "bicycle"},
    {"category_id": 25, "name": "bird"},
    {"category_id": 31, "name": "boat"},
    {"category_id": 33, "name": "book"},
    {"category_id": 34, "name": "bottle"},
    {"category_id": 44, "name": "building"},  # 10
    {"category_id": 45, "name": "bus"},
    {"category_id": 46, "name": "cabinet"},
    {"category_id": 59, "name": "car"},
    {"category_id": 65, "name": "cat"},
    {"category_id": 68, "name": "ceiling"},
    {"category_id": 72, "name": "chair"},
    {"category_id": 80, "name": "cloth"},
    {"category_id": 85, "name": "computer"},
    {"category_id": 98, "name": "cow"},
    {"category_id": 104, "name": "cup"},  # 20
    {"category_id": 105, "name": "curtain"},
    {"category_id": 113, "name": "dog"},
    {"category_id": 115, "name": "door"},
    {"category_id": 144, "name": "fence"},
    {"category_id": 158, "name": "floor"},
    {"category_id": 159, "name": "flower"},
    {"category_id": 162, "name": "food"},
    {"category_id": 187, "name": "grass"},
    {"category_id": 189, "name": "ground"},
    {"category_id": 207, "name": "horse"},  # 30
    {"category_id": 220, "name": "keyboard"},
    {"category_id": 232, "name": "light"},
    {"category_id": 258, "name": "motorbike"},
    {"category_id": 259, "name": "mountain"},
    {"category_id": 260, "name": "mouse"},
    {"category_id": 284, "name": "person"},
    {"category_id": 295, "name": "plate"},
    {"category_id": 296, "name": "platform"},
    {"category_id": 308, "name": "pottedplant"},
    {"category_id": 324, "name": "road"},  # 40
    {"category_id": 326, "name": "rock"},
    {"category_id": 347, "name": "sheep"},
    {"category_id": 349, "name": "shelves"},
    {"category_id": 354, "name": "sidewalk"},
    {"category_id": 355, "name": "sign"},
    {"category_id": 360, "name": "sky"},
    {"category_id": 366, "name": "snow"},
    {"category_id": 368, "name": "sofa"},
    {"category_id": 397, "name": "diningtable"},
    {"category_id": 415, "name": "track"},  # 50
    {"category_id": 416, "name": "train"},
    {"category_id": 420, "name": "tree"},
    {"category_id": 424, "name": "truck"},
    {"category_id": 427, "name": "tvmonitor"},
    {"category_id": 440, "name": "wall"},
    {"category_id": 445, "name": "water"},
    {"category_id": 454, "name": "window"},
    {"category_id": 458, "name": "wood"},
]


def _get_pascal_context_59_meta():
    assert len(PASCAL_59_CATEGORIES) == 59, len(PASCAL_59_CATEGORIES)
    stuff_classes = [k["name"] for k in PASCAL_59_CATEGORIES]
    ret = {"stuff_classes": stuff_classes}
    return ret


def load_pascal_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg", phase="val"):
    """
    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.
    """
    assert phase == "val"

    img_path = os.path.abspath(os.path.join(image_root, "..", f"{phase}.txt"))
    
    img_list = []
    with open(img_path, "r") as f:
        img_list = f.readlines()
    img_list = [name[:-1] for name in img_list]  # '\n'

    dataset_dicts = []
    for img in img_list:
        record = {
           "file_name": os.path.join(image_root, f"{img}.{image_ext}"),
            "sem_seg_file_name": os.path.join(gt_root, f"{img}.{gt_ext}"),
        }
        dataset_dicts.append(record)

    return dataset_dicts


def register_pascal_context_59(root):
    root = os.path.join(root, "pascal_context_full")
    meta = _get_pascal_context_59_meta()
    for name in ["train", "val"]:
        image_dir = os.path.join(root, "images")
        gt_dir = os.path.join(root, "labels_59")
        name = f"pascal_context_59_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_pascal_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_pascal_context_59(_root)  # 59
