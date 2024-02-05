# Explainable-YOLOv8
Visualize the low-level outputs of YOLOv8 to analyze and understand the areas where our model focuses. Specifically, illustrate which anchor points are activated to predict bounding boxes.

## 🧍
![image](https://github.com/developer0hye/Explainable-YOLOv8/assets/35001605/e464ac44-c92b-4c06-abe0-557c0d773ef8)
## 👔
![image](https://github.com/developer0hye/Explainable-YOLOv8/assets/35001605/efe440f3-066b-43b3-b2fc-de7b4dbb1fc3)
## 🚌
![image](https://github.com/developer0hye/Explainable-YOLOv8/assets/35001605/bc3ff9be-503a-4825-b681-f4f9e963e843)


Green points indicate the areas where YOLOv8 focuses, with brighter green representing a higher confidence score.

Green bounding boxes represent those with high confidence scores; these boxes have not been processed with Non-Maximum Suppression (NMS). 

Each arrow represents the predicted left, top, right, and bottom (LTRB) distances from the anchor points.

I am aware that my plotting method is not :thumbsup:. Feel free to modify the code to enhance the quality of the figure.


## Install
```bash
pip install ultralytics==8.1.8
git clone https://github.com/developer0hye/Explainable-YOLOv8.git
cd Explainable-YOLOv8
```

## Run
```
python visualize.py --model {your model}.pt --source {your data}
```

Execute the command below to generate the image as shown in this README file

```
cd Explainable-YOLOv8
python visualize.py --model yolov8m.pt --source ./ --class_id 0
python visualize.py --model yolov8m.pt --source ./ --class_id 27
python visualize.py --model yolov8m.pt --source ./ --class_id 5
```
