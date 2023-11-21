# coding: utf-8
import os
import os.path as osp
import re
from glob import glob
import subprocess
import random
from tqdm import tqdm

# ckpts = glob("/checkpoint/avatar/wenj/oss_release/*/*/*/*/*.pt")
ckpts = glob("/checkpoint/avatar/julietamartinez/oss_release/*/*/*/*/*.pt")

# print(ckpts)
numbered_ckpts = list(filter(lambda x:re.search("aeparams_[0-9]+.pt$", x) is not None, ckpts))
# print(numbered_ckpts)
# quitz()
its = [int(osp.basename(fn).split(".")[0].split("_")[1]) for fn in numbered_ckpts]

deleted_ckpts = []
for it, fn in zip(its, numbered_ckpts):
   print(it, fn)
   if it % 20_000 or (it < 1_000 and it % 100 != 0):
      deleted_ckpts.append(fn)

print(deleted_ckpts)
print(f"deleting {len(deleted_ckpts)} out of {len(ckpts)} checkpoints")

pids = []
for fn in tqdm(deleted_ckpts):
   pids.append(subprocess.Popen(["rm", fn]))

for p in pids:
   p.communicate()
