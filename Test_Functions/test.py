from ultralytics import YOLO

model = YOLO("/Users/sameersimon/Local Documents/MMA_3001_Project/results/best.pt")

metrics = model.val(
    data="/Users/sameersimon/Local Documents/MMA_3001_Project/Pork Rasher Error-Packaging-.v1i.yolov8/data.yaml",
    split="test",
    imgsz=640,
    plots=True,
    project="runs/test",
    name="yolov8n_v1_test"
)

print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP@50:", metrics.box.map50)
print("mAP@50-95:", metrics.box.map)
print("Per-class mAP@50-95:", metrics.box.maps)
print("Speed breakdown (ms/image):", metrics.speed)