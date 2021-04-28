# -*- coding: utf-8 -*-
# Copyright (c) 2021 JSON LEE, lijiansong@ict.ac.cn.
# All rights reserved.

import argparse
import numpy as np
from datetime import datetime
import time

import model as vgg_model

import megengine as mge
import megengine.autodiff as ad
import megengine.module as M
import megengine.functional as F
import megengine.optimizer as optim

parser = argparse.ArgumentParser(description="MegEngine VGG Training")
parser.add_argument(
    "--steps",
    default=10,
    type=int,
    help="number of total steps to run (default: 10)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    metavar="SIZE",
    default=256,
    type=int,
    help="batch size for single GPU (default: 256)",
)
parser.add_argument(
    "--enable-dtr",
    dest="enable_dtr",
    action="store_true",
    help="Enable DTR")
parser.add_argument(
    "--memory-budget",
    dest="mem_budget",
    default=5,
    type=float,
    help="memory budget for DTR, measured in GB (default: 5.0)",
)

args = parser.parse_args()

if args.enable_dtr:
    from megengine.utils.dtr import DTR
    ds = DTR(memory_budget=int(args.mem_budget*1024**3))

model = vgg_model.VGG()
print(model)
batch_size = args.batch_size
image = mge.tensor(np.random.random((batch_size, 3, 32, 32)))
label = mge.tensor(np.random.randint(100, size=(batch_size,)))

gm=ad.GradManager().attach(model.parameters())
opt=optim.SGD(model.parameters(), lr=0.0125, momentum=0.9, weight_decay=1e-4)

# miliseconds
print(datetime.now().timetz())
time_list = []
cur_time = int(round(time.time()*1000))
for i in range(args.steps):
    with gm:
        logits=model(image)
        loss=F.nn.cross_entropy(logits, label)
        gm.backward(loss)
        opt.step().clear_grad()

        next_time = int(round(time.time()*1000))
        time_list.append(next_time - cur_time)
        cur_time = next_time

        print("iter = {}, loss = {}".format(i+1, loss.numpy()))

print('throughput: {} ms!!!'.format(np.average(np.array(time_list))))
