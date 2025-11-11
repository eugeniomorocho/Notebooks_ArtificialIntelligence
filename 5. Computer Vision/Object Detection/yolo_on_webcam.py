from ultralytics import YOLO
model = YOLO('yolo11n.pt')

results = model.predict(
    source=0, # source=0 uses the default webcam, use source=1 for an external camera
    show=True, # show=True to display the video feed with detections in a window 
    stream=True, # stream=True for real-time processing of video frames
    imgsz=256, # resize frames to 256x256 for faster processing and lower resource usage
    vid_stride=3, # process every 3rd frame to reduce CPU/GPU load
    save=False # do not save the video output (this saves disk space and speeds up processing)
    #verbose=False # stops printing per-frame logs
    #conf=0.3, # confidence threshold of 30% (default is 0.25). Adjust as needed.
)