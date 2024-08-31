import os
import pickle
import math
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing, listdir_nohidden

def read_and_split_data(image_dir, p_trn=0.5, p_val=0.2, ignored=[], new_cnames=None):
    categories = listdir_nohidden(image_dir)
    categories = [c for c in categories if c not in ignored]
    categories.sort()
     
    p_tst = 1 - p_trn - p_val
    print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

def save_split(train, val, test, filepath, path_prefix):
    def _extract(items):
        out = []
        for item in items:
            impath = item.impath
            label = item.label
            classname = item.classname
            impath = impath.replace(path_prefix, "")
            if impath.startswith("/"):
                impath = impath[1:]
            out.append((impath, label, classname))
        return out

    train = _extract(train)
    val = _extract(val)
    test = _extract(test)

    split = {"train": train, "val": val, "test": test}

    write_json(split, filepath)
    print(f"Saved split to {filepath}")