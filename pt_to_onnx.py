from ultralytics import YOLO

model = YOLO("best.pt")  # load a pretrained model (recommended for training)
path = model.export(format="onnx",  imgsz=[640,640], opset=12)  # export the model to ONNX format