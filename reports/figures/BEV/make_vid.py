import cv2
import os
import glob

def make_video_from_images(folder, output_path, fps=10):
    images = sorted(
        glob.glob(os.path.join(folder, "bev_overlay_*.png")),
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    if len(images) == 0:
        raise ValueError("No images found")

    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        video.write(frame)

    video.release()

# usage
make_video_from_images(
    folder='reports/figures/BEV/video',
    output_path=os.path.join('reports/figures/BEV/video', "bev_video.mp4"),
    fps=3
)