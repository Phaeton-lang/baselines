# MegEngine DTR

DTR功能的支持目前还没有merge到主分支，源码见release-1.4分支: <https://github.com/MegEngine/MegEngine/tree/release-1.4/>
DTR功能实现见该分支的此次提交:
dynamic sublinear
```
$ git show fe99cdc794bfa77f33e96b99bb1f8eaa4e3c3bee
```

```
$ pip3 install megengine==1.4.0rc1 --no-cache-dir
```
<https://github.com/MegEngine/MegEngine/releases/tag/v1.4.0-rc1>

see here: <https://github.com/MegEngine/MegEngine/issues/37>

目前用`mge.get_mem_status_bytes()`来获取GPU显存的占用情况，这种方式不准确！跟插桩的位置相关！！！(对部分模型而言，可能小于实际的显存占用，可以综合考虑查看nvidia-smi)

```
total, free = mge.get_mem_status_bytes()
print('iter = {}, used bytes(/MB) = {}'.format(i+1, float(total - free)/1024.0/1024.0))
```
