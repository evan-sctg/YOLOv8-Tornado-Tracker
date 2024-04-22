from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")
results = model.train(
   data='tornados.yaml',
   imgsz=640,
   epochs=500,
   batch=8,
   name='yolov8m-seg_10e'
)
