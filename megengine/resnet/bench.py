import numpy as np
import model as resnet_model
import megengine as mge
import megengine.autodiff as ad
import megengine.functional as F
import megengine.optimizer as optim

from megengine.utils.dtr import DTR
dtr = DTR(memory_budget=5*1024**3)

batch_size = 64
image = mge.tensor(np.random.random((batch_size, 3, 224, 224)))
label = mge.tensor(np.random.randint(100, size=(batch_size,)))
model = resnet_model.__dict__["resnet50"]()

gm=ad.GradManager().attach(model.parameters())
opt=optim.SGD(model.parameters(), lr=0.0125, momentum=0.9, weight_decay=1e-4)

n_iter = 20

for i in range(n_iter):
    with gm:
        logits=model(image)
        loss=F.nn.cross_entropy(logits, label)
        gm.backward(loss)
        opt.step().clear_grad()
        print("iter = {}, loss = {}".format(i+1, loss.numpy()))
