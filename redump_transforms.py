"""Redump fom very chonky transforms with metadata to just the parts we need"""

import os
import shutil
import torch
import pickle
import pathlib
import pandas as pd
from data.nr_dataset import MugsyCapture

def main():
    captures_df = pd.read_csv(pathlib.Path(__file__).parent / "217_ids.csv", dtype=str)

    captures = []
    for _, row in captures_df.iterrows():
        capture = MugsyCapture(mcd=row['mcd'], mct=row['mct'], sid=row['sid'])
        if capture.sid.lower()[:3] in {"ajd", "bnp", "zse", "apb"}:
            continue
        captures.append(capture)

    # dataset = MultiCaptureDataset(captures)

    for capture in captures:
        print(capture)

        fsrc = f"/uca/julieta/cache/m--{capture.mcd}--{capture.mct}--{capture.sid}--GHS/cambyte_transforms.pkl"
        fdst = f"/uca/julieta/cache/m--{capture.mcd}--{capture.mct}--{capture.sid}--GHS/cambyte_transforms.bak.pkl"
        shutil.move(fsrc, fdst)

        with open(fdst, "rb") as handle:
            chunky = pickle.load(handle)

        krt_fname = f"/uca/julieta/cache/m--{capture.mcd}--{capture.mct}--{capture.sid}--GHS/krt_dict.pkl"
        with open(krt_fname, "wb") as handle:
            pickle.dump(chunky["cambyte_metadata"]["krt_dict"], handle)

        with open(fsrc, "wb") as handle:
            pickle.dump(chunky["cambyte_transforms"], handle)



if __name__ == "__main__":
    main()
