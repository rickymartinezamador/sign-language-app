import pickle
import json
import numpy as np

SIGN_MODEL_PATH = "sign_model.pkl"
OUTPUT_PATH = "model.json"

with open(SIGN_MODEL_PATH, "rb") as f:
    model = pickle.load(f)

classes = list(model.classes_)

trees_json = []

for estimator in model.estimators_:
    tree = estimator.tree_
    nodes = []

    for i in range(tree.node_count):
        feature = int(tree.feature[i])
        threshold = float(tree.threshold[i])
        left = int(tree.children_left[i])
        right = int(tree.children_right[i])

        is_leaf = feature == -2

        node = {
            "feature": feature,
            "threshold": threshold,
            "left": left,
            "right": right,
        }

        if is_leaf:
            counts = tree.value[i][0]
            total = counts.sum()
            probs = (counts / total).tolist() if total > 0 else counts.tolist()
            node["probs"] = probs

        nodes.append(node)

    trees_json.append(nodes)

output = {
    "classes": classes,
    "trees": trees_json
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f)

print(f"Exported {len(trees_json)} trees covering {len(classes)} letters to {OUTPUT_PATH}")

import os
size_kb = os.path.getsize(OUTPUT_PATH) / 1024
print(f"File size: {size_kb:.1f} KB")