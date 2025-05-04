from ultralytics import YOLO
import os
import shutil

model = YOLO("/yolov11_muzdogs.pt")
RESULTS_PATH = "/results"

def predict_image(image_path):
    results = model(image_path)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    output_path = os.path.join(RESULTS_PATH, os.path.basename(image_path))
    
    objects_count = 0
    for r in results:
        r.save(filename=output_path)
        objects_count += len(r.boxes)
    
    return os.path.basename(output_path), objects_count

def predict_video(video_path):
    os.makedirs(RESULTS_PATH, exist_ok=True)

    results = model.predict(
        source=video_path,
        save=True,
        project=RESULTS_PATH,
        name="predict",
        exist_ok=True
    )

    video_name = os.path.basename(video_path)
    saved_video_path = os.path.join(RESULTS_PATH, "predict", video_name)
    final_path = os.path.join(RESULTS_PATH, video_name)
    
    objects_count = 0
    for r in results:
        objects_count += len(r.boxes)
    
    if os.path.exists(saved_video_path):
        shutil.move(saved_video_path, final_path)

    return os.path.basename(final_path), objects_count
