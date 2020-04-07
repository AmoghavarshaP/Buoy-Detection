# Buoy-Detection
The project is based on the concept of color segmentation using Gaussian Mixture Models and
Expectation Maximization techniques. The video sequence we have has been captured underwater
and shows three buoys of different colors, namely yellow, orange and green. They are almost circular in shape
and are distinctly colored. However, conventional segmentation techniques involving color thresholding will
not work well in such an environment, since noise and varying light intensities will render any hard-coded
thresholds ineffective. In such a scenario, we will “learn” the color distributions of the buoys and use that 
learned model to segment them. The output of the project shows a tight segmentation of each buoy for
the entire video sequence by applying a tight contour (in the respective color of the buoy being segmented)
around each buoy.

## Authors
- Pruthvi Sanghavi
- Naman Gupta
- Amoghavarsha Prassana

## Dependencies
- Numpy ```pip install numpy```
- OpenCV ```pip install opencv-python```
- Matplotlib ```pip install matplotlib```

## Run Instructions
Open the terminal and type the following commands
```
cd <workspace>
git clone https://github.com/AmoghavarshaP/Buoy-Detection.git
cd Buoy-Detection
```
- For training (for red buoy)
```
python3 segRed_multi.py
```
- For detection
```
python3 detection3D.py
```
## Results
- Link to Results : [Results Drive Link](https://drive.google.com/drive/u/0/folders/17eO_HZwCzxNqrd-DXKExQSckOnobPJWr)
