# DMA Loss


## Prerequisites
Prepare running environment:
```
pip install -r requirements.txt
```


## Datasets
*  We used the following four public benchmark datasets: [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/), [Cars-196](https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch), [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/) and [In-shop Clothes Retrieval](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html).
* Extract the downloaded datasets into the `./data/` directory. For Cars-196, place the files in a `./data/cars196/` directory instead.


## Training

For CUB-200-2011:
```bash
python code/train.py --gpu-id 0 \
                --tau 0.2 \
                --nb_c 10 \
                --lr 1e-4 \
                --dataset cub \
                --bn-freeze 1 \
                --lr-decay-step 5
```

For Cars-196:
```bash
python code/train.py --gpu-id 0 \
                --tau 0.2 \
                --nb_c 10 \
                --lr 1e-4 \
                --dataset cars \
                --bn-freeze 1 \
                --lr-decay-step 10
```

For Stanford Online Products:
```bash
python code/train.py --gpu-id 0 \
                --tau 0 \
                --nb_c 2 \
                --lr 6e-4 \
                --dataset SOP \
                --bn-freeze 0 \
                --lr-decay-step 10 \
                --lr-decay-gamma 0.25
```

For In-Shop Clothes Retrieval:
```bash
python code/train.py --gpu-id 0 \
                --tau 0 \
                --nb_c 2 \
                --lr 6e-4 \
                --dataset Inshop \
                --bn-freeze 0 \
                --lr-decay-step 10 \
                --lr-decay-gamma 0.25
```