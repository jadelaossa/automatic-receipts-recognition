from ultralytics import YOLO


# load the model
model = YOLO("runs/detect/train2/weights/best.pt")

# run batched inference on a list of images
results = model([
    "data/digital/mercadona/test/images/mercadona_00005_png.rf.e389c6b7c2ae4aae357063fe1ad08335.jpg",
    "data/digital/mercadona/test/images/mercadona_00009_png.rf.5731b6ae78e6a94f3fdcaf2a93442186.jpg",
    "data/digital/mercadona/test/images/mercadona_00024_png.rf.5f9771a1514c4c83e3f985ab0845c5c7.jpg"
])

for result in results:
    boxes = result.boxes    # boxes object for bounding box outputs
    masks = result.masks    # masks object for segmentation masks outputs
    keypoints = result.keypoints    # keypoints object for pose outputs
    probs = result.probs    # probs object for classification outputs
    obb = result.obb    # oriented boxes object for OBB outputs
    result.show()   # display to screen