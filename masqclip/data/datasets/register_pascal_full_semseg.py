# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog

PASCAL_459_CATEGORIES = [
    "accordion", "aeroplane", "air conditioner", "antenna", "artillery", "ashtray", 
    "atrium", 
    "baby carriage", "bag", "ball", "balloon", "bamboo weaving", "barrel", "baseball bat", 
    "basket", "basketball backboard", "bathtub", "bed", "bedclothes", "beer", "bell", 
    "bench", "bicycle", "binoculars", "bird", "bird cage", "bird feeder", "bird nest", 
    "blackboard", "board", "boat", "bone", "book", "bottle", "bottle opener", "bowl", 
    "box", "bracelet", "brick", "bridge", "broom", "brush", "bucket", "building", 
    "bus", 
    "cabinet", "cabinet door", "cage", "cake", "calculator", "calendar", "camel", 
    "camera", "camera lens", "can", "candle", "candle holder", "cap", "car", "card", 
    "cart", "case", "casette recorder", "cash register", "cat", "cd", "cd player",
    "ceiling", "cell phone", "cello", "chain", "chair", "chessboard", "chicken", 
    "chopstick", "clip", "clippers", "clock", "closet", "cloth", "clothes tree", 
    "coffee", "coffee machine", "comb", "computer", "concrete", "cone", "container", 
    "control booth", "controller", "cooker", "copying machine", "coral", "cork", 
    "corkscrew", "counter", "court", "cow", "crabstick", "crane", "crate", "cross",
    "crutch", "cup", "curtain", "cushion", "cutting board", 
    "dais", "disc", "disc case", "dishwasher", "dock", "dog", "dolphin", "door", 
    "drainer", "dray", "drink dispenser", "drinking machine", "drop", "drug", 
    "drum", "drum kit", "duck", "dumbbell", 
    "earphone", "earrings", "egg", "electric fan", "electric iron", "electric pot",
    "electric saw", "electronic keyboard", "engine", "envelope", "equipment", "escalator", 
    "exhibition booth", "extinguisher", "eyeglass", 
    "fan", "faucet", "fax machine", "fence", "ferris wheel", "fire extinguisher", 
    "fire hydrant", "fire place", "fish", "fish tank", "fishbowl", "fishing net", 
    "fishing pole", "flag", "flagstaff", "flame", "flashlight", "floor", "flower", 
    "fly", "foam", "food", "footbridge", "forceps", "fork", "forklift", "fountain", 
    "fox", "frame", "fridge", "frog", "fruit", "funnel", "furnace", 
    "game controller", "game machine", "gas cylinder", "gas hood", "gas stove", 
    "gift box", "glass", "glass marble", "globe", "glove", "goal", "grandstand", 
    "grass", "gravestone", "ground", "guardrail", "guitar", "gun", 
    "hammer", "hand cart", "handle", "handrail", "hanger", "hard disk drive", "hat", 
    "hay", "headphone", "heater", "helicopter", "helmet", "holder", "hook", "horse", 
    "horse-drawn carriage", "hot-air balloon", "hydrovalve", 
    "ice", "inflator pump", "ipod", "iron", "ironing board", 
    "jar", 
    "kart", "kettle", "key", "keyboard", "kitchen range", "kite", "knife", "knife block", 
    "ladder", "ladder truck", "ladle", "laptop", "leaves", "lid", "life buoy", 
    "light", "light bulb", "lighter", "line", "lion", "lobster", "lock", 
    "machine", "mailbox", "mannequin", "map", "mask", "mat", "match book", "mattress", 
    "menu", "metal", "meter box", "microphone", "microwave", "mirror", "missile", 
    "model", "money", "monkey", "mop", "motorbike", "mountain", "mouse", "mouse pad", 
    "musical instrument", 
    "napkin", "net", "newspaper", 
    "oar", "ornament", "outlet", "oven", "oxygen bottle", 
    "pack", "pan", "paper", "paper box", "paper cutter", "parachute", "parasol", 
    "parterre", "patio", "pelage", "pen", "pen container", "pencil", "person", 
    "photo", "piano", "picture", "pig", "pillar", "pillow", "pipe", "pitcher", 
    "plant", "plastic", "plate", "platform", "player", "playground", "pliers", 
    "plume", "poker", "poker chip", "pole", "pool table", "postcard", "poster", 
    "pot", "pottedplant", "printer", "projector", "pumpkin", 
    "rabbit", "racket", "radiator", "radio", "rail", "rake", "ramp", "range hood", 
    "receiver", "recorder", "recreational machines", "remote control", "road", 
    "robot", "rock", "rocket", "rocking horse", "rope", "rug", "ruler", "runway", 
    "saddle", "sand", "saw", "scale", "scanner", "scissors", "scoop", "screen", 
    "screwdriver", "sculpture", "scythe", "sewer", "sewing machine", "shed", "sheep", 
    "shell", "shelves", "shoe", "shopping cart", "shovel", "sidecar", "sidewalk", 
    "sign", "signal light", "sink", "skateboard", "ski", "sky", "sled", "slippers", 
    "smoke", "snail", "snake", "snow", "snowmobiles", "sofa", "spanner", "spatula", 
    "speaker", "speed bump", "spice container", "spoon", "sprayer", "squirrel", 
    "stage", "stair", "stapler", "stick", "sticky note", "stone", "stool", "stove", 
    "straw", "stretcher", "sun", "sunglass", "sunshade", "surveillance camera", 
    "swan", "sweeper", "swim ring", "swimming pool", "swing", "switch", 
    "table", "tableware", "tank", "tap", "tape", "tarp", "telephone", "telephone booth", 
    "tent", "tire", "toaster", "toilet", "tong", "tool", "toothbrush", "towel", 
    "toy", "toy car", "track", "train", "trampoline", "trash bin", "tray", "tree", 
    "tricycle", "tripod", "trophy", "truck", "tube", "turtle", "tvmonitor", "tweezers", 
    "typewriter", 
    "umbrella", "unknown", 
    "vacuum cleaner", "vending machine", "video camera", "video game console", 
    "video player", "video tape", "violin", 
    "wakeboard", "wall", "wallet", "wardrobe", "washing machine", "watch", "water", 
    "water dispenser", "water pipe", "water skate board", "watermelon", "whale", 
    "wharf", "wheel", "wheelchair", "window", "window blinds", "wineglass", "wire", 
    "wood", "wool",
]


def _get_pascal_context_459_meta():
    stuff_classes = PASCAL_459_CATEGORIES
    assert len(stuff_classes) == 459, len(stuff_classes)
    ret = {"stuff_classes": stuff_classes}
    return ret


def load_pascal_full_sem_seg(gt_root, image_root, gt_ext="png", image_ext="jpg", phase="val"):
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


def register_pascal_context_459(root):
    root = os.path.join(root, "pascal_context_full")
    meta = _get_pascal_context_459_meta()
    for name in ["train", "val"]:
        image_dir = os.path.join(root, "images")
        gt_dir = os.path.join(root, "labels_459")
        name = f"pascal_context_459_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_pascal_full_sem_seg(y, x, gt_ext="tif", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=65535,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_pascal_context_459(_root)  # 459
