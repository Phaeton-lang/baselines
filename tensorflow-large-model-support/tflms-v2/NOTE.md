# NOTE

tensorflow 2.1.3 has bugs, it will frozen the training in RTX 3090,

for tensorflow-gpu 2.2.0,

```
$ conda config --prepend channels \
https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda-early-access/
$ conda create --name wmlce_env python=3.6
$ conda activate wmlce_env
$ conda install tensorflow-gpu=2.2.0
```

Update cuDNN of conda, see here: <https://stackoverflow.com/questions/55256671/how-to-install-latest-cudnn-to-conda>
