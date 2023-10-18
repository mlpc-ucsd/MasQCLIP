import json
import os

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES


val_path = "coco/annotations/instances_val2017.json"
train_path = "coco/annotations/instances_train2017.json"
outdir = "coco/annotations_ov_split_48_17"
if not os.path.exists(outdir):
    os.makedirs(outdir)

seen_classes_names = ('toilet', 'bicycle', 'apple', 'train', 'laptop', 'carrot', 'motorcycle', 'oven', 'chair', 'mouse', 'boat', 'kite', 'sheep', 'horse', 'sandwich', 'clock', 'tv', 'backpack', 'toaster', 'bowl', 'microwave', 'bench', 'book', 'orange', 'bird', 'pizza', 'fork', 'frisbee', 'bear', 'vase', 'toothbrush', 'spoon', 'giraffe', 'handbag', 'broccoli', 'refrigerator', 'remote', 'surfboard', 'car', 'bed', 'banana', 'donut', 'skis', 'person', 'truck', 'bottle', 'suitcase', 'zebra')
unseen_classes_names = ('umbrella', 'cow', 'cup', 'bus', 'keyboard', 'skateboard', 'dog', 'couch', 'tie', 'snowboard', 'sink', 'elephant', 'cake', 'scissors', 'airplane', 'cat', 'knife')
instances65_classes_names = seen_classes_names + unseen_classes_names

assert len(seen_classes_names) == 48
assert len(unseen_classes_names) == 17
assert len(instances65_classes_names) == 65

split_dict = {
    "seen48": seen_classes_names,
    "unseen17": unseen_classes_names,
    "instances65": instances65_classes_names
}

for set_type in ["train", "val"]:
    input_json = "coco/annotations/instances_%s2017.json" % set_type
    with open (input_json) as f:
        content = json.load(f)
        for split_key in split_dict.keys():
            images_id_set = set()
            class_names = split_dict[split_key]
            class_ids = set()
            for cat in COCO_CATEGORIES:
                if cat["name"] in class_names:
                    class_ids.add(cat['id'])

            out_json_path = os.path.join(outdir, "instances_%s2017_%s.json" % (set_type, split_key))
            print("[output]", out_json_path)
            out_content = dict()

            out_content["info"] = content['info']
            out_content["licenses"] = content["licenses"]
            print("write info and licenses")

            out_content['categories'] = []
            for cat in content["categories"]:
                # print("cat", cat)
                if cat["id"] in class_ids:
                    out_content["categories"].append(cat)
            print("categories", split_key, len(out_content["categories"]))

            out_content["annotations"] = []
            for anno in content["annotations"]:
                if anno["category_id"] in class_ids:
                    images_id_set.add(anno['image_id'])
                    out_content["annotations"].append(anno)
            print("annotations: reduce %d to %d " %(len(content["annotations"]), len(out_content["annotations"])))

            out_content["images"] = []
            for img in content["images"]:
                if img["id"] in images_id_set:
                    out_content["images"].append(img)
            print("images: reduce %d to %d \n" %(len(content["images"]), len(out_content["images"])))


            with open(out_json_path, "w") as fout:
                json.dump(out_content, fout)
