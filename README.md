#### Reference

[https://github.com/amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)

#### What we do

We adapted the code to pytorch version 0.4.0. And some bug was fixed.

#### Still have some problems

```shell
Loading base network...
Initializing weights...
Loading the dataset...
Training SSD on: VOC2007
Using the specified args:
Namespace(basenet='vgg16_reducedfc.pth', batch_size=4, dataset='VOC', dataset_root='C:\\Users\\jimch/Documents/Github/ssd/data/VOCdevkit/', gamma=0.1, lr=0.001, momentum=0.9, num_workers=4, resume=None, save_folder='weights/', start_iter=0, visdom=False, weight_decay=0.0005)
timer: 7.9134 sec.
iter 0 || Loss: 27.4027 || timer: 0.1308 sec.
iter 10 || Loss: 14.9920 || timer: 0.1609 sec.
iter 20 || Loss: 20.7590 || timer: 0.1654 sec.
iter 30 || Loss: 20.2252 || timer: 0.1409 sec.
iter 40 || Loss: 15.1791 || timer: 0.1519 sec.
iter 50 || Loss: 14.1190 || timer: 0.1479 sec.
iter 60 || Loss: 13.9990 || timer: 0.1509 sec.
iter 70 || Loss: 120128.8672 || Traceback (most recent call last):
  File "train.py", line 224, in <module>
    train()
  File "train.py", line 149, in train
    loss_l, loss_c = criterion(out, targets)
  File "C:\Users\jimch\Anaconda3\lib\site-packages\torch\nn\modules\module.py", line 491, in __call__
    result = self.forward(*input, **kwargs)
  File "C:\Users\jimch\Documents\GitHub\ssd\layers\modules\multibox_loss.py", line 99, in forward
    _, loss_idx = loss_c.sort(1, descending=True)
RuntimeError: merge_sort: failed to synchronize: an illegal memory access was encountered
```

corresponding issue: [https://github.com/amdegroot/ssd.pytorch/issues/120](https://github.com/amdegroot/ssd.pytorch/issues/120)