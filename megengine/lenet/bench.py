# -*- coding: utf-8 -*-
import argparse
import numpy as np
from datetime import datetime
import time

import model as lenet_model

import megengine as mge
import megengine.autodiff as ad
import megengine.module as M
import megengine.functional as F
import megengine.optimizer as optim

parser = argparse.ArgumentParser(description="MegEngine LeNet Training")
parser.add_argument(
    "--steps",
    default=10,
    type=int,
    help="number of total steps to run (default: 10)",
)
parser.add_argument(
    "--hw",
    default=32,
    type=int,
    choices=[32, 224],
    help="height width of input image (default: 32)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    metavar="SIZE",
    default=256,
    type=int,
    help="batch size for single GPU (default: 64)",
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
    type=int,
    help="memory budget for DTR, measured in GB (default: 5)",
)

args = parser.parse_args()

if args.enable_dtr:
    from megengine.utils.dtr import DTR
    ds = DTR(memory_budget=args.mem_budget*1024**3)

batch_size = args.batch_size
image = mge.tensor(np.random.random((batch_size, 1, args.hw, args.hw)))
label = mge.tensor(np.random.randint(100, size=(batch_size,)))

if args.hw == 32:
    model = lenet_model.LeNet32x32()
elif args.hw == 224:
    model = lenet_model.LeNet224x224()
print(model)
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
        total, free = mge.get_mem_status_bytes()
        print('iter = {}, used bytes(/MB) = {}'.format(i+1, float(total - free)/1024.0/1024.0))
        opt.step().clear_grad()

        next_time = int(round(time.time()*1000))
        time_list.append(next_time - cur_time)
        cur_time = next_time

        print("iter = {}, loss = {}".format(i+1, loss.numpy()))

print('throughput: {} ms!!!'.format(np.average(np.array(time_list))))
