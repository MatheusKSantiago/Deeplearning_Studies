from ultralytics import YOLO
import os
model = YOLO("yolo11n-cls.pt")
model.load("runs\\classify\\train\\weights\\best.pt")
dataset = os.path.join("HAM10000_ESTRUTURADO")

model.train(data=dataset, epochs=60)