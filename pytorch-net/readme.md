```
python train_jinnan.py --pus 0,1,2,3
```

![1569948311882](img/1569948311882.png)

If you want to start training from a break point **checkpoint**：

for example

```
python train_jinnan.py --pus 0,1,2,3 --resume_from model/epoch_10.pth
```

------

```
python test_jinnan.py
```

![1569948488703](img/1569948488703.png)

