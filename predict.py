from ultralytics import YOLO


# load the model
model = YOLO("runs/detect/train/weights/best.pt")

# run batched inference on a list of images
results = model([
    "data/test/images/receipt_00024_jpeg.rf.3f9f40393acc174489b78c7709d47cf6.jpg",
    "data/test/images/receipt_00077_jpeg.rf.037ce073d1740073ec73836902ca550c.jpg",
    "data/test/images/receipt_00135_jpg.rf.bab58cb3dd7e7e617f57dfa554ea9601.jpg",
    "data/test/images/receipt_00108_jpeg.rf.1d531dd3e153e66c5396ab45fdd10ea8.jpg"
])

for result in results:
    boxes = result.boxes    # boxes object for bounding box outputs
    masks = result.masks    # masks object for segmentation masks outputs
    keypoints = result.keypoints    # keypoints object for pose outputs
    probs = result.probs    # probs object for classification outputs
    obb = result.obb    # oriented boxes object for OBB outputs
    result.show()   # display to screen