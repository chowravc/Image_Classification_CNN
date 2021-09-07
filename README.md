# Image Classification CNN

## Training information

Create a folder `data/` and put your data set at `data/<data-set>/`.
The format of the data set is:
```
<data-set>
|-- test/
    |-- images/
    |-- labels/
|-- train/
    |-- images/
    |-- labels/
```
The test set and train set must contain images called `<filename>.png` in `images/` and associated labels (numpy arrays to txt) `<filename>.txt` of the same size in the corresponding directory `labels/`.
 
After creating this dataset, open `train.py`, and put path to dataset `data/<data-set>/` in line 37. Other values such as batch size, learning rate and model can be chosen here.

You can also choose the number of epochs directly in the training loop.

For the model 'CNN', a train image size of 256x256 is expected.

Finally, run `train.py` with:
```
!python train.py
```

Training results will get stored `runs/train/<exp>/` and one weight will be stored every epoch in `checkpoints/`. The last epoch weights and best epoch weights will be stored in `weights/`. An in-depth look at the training is stored to `results.txt`.

## Detection information

All images to be detected must go to new directory `data/<new-image-directory>/<your-images>.<ext>`.

After putting images here, run:
```
!python -W ignore detect.py --w <path-to-trained-weight> --src data/<your-images>/
```

Currently, the only model choice is CNN. Make sure your weight was trained for the right model.

Example weight path:
```
runs/train/exp1/weights/best.pth
```

Detection results will be stored in `runs/detect/<exp>/`, with numpy arrays as txt in `labels/`.
