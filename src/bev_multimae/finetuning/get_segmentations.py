import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import matplotlib.pyplot as plt

def plot_img_and_seg(img, mask):
    if img is None or mask is None:
        print("Missing image or mask")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy()
    overlay[mask == 1] = [255, 0, 0]
    blended = cv2.addWeighted(img_rgb, 0.5, overlay, 0.5, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Segmentation")
    axes[2].imshow(blended)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig('reports/figures/finetuning/test_plot.png')

def extract_img_and_seg(input_path, img_topic, seg_topic):
    img = None
    seg_mask = None

    with open(input_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        for schema, channel, message, ros_msg in reader.iter_decoded_messages(
            topics=[img_topic, seg_topic]
        ):

            if channel.topic == img_topic:
                img_np = np.frombuffer(ros_msg.data, dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            elif channel.topic == seg_topic and img is not None:
                H, W = img.shape[:2]
                seg_mask = np.zeros((H, W), dtype=np.uint8)

                for ann in ros_msg.points:
                    for p in ann.points:
                        x = int(p.x)
                        y = int(p.y)
                        if 0 <= x < W and 0 <= y < H:
                            cv2.circle(seg_mask, (x, y), 4, 1, -1)

            if img is not None and seg_mask is not None:
                break

    return img, seg_mask

def main():

    input_path = 'data/raw/bags/evt_0dpi1vKWJgizY6Vy.mcap'

    img, mask = extract_img_and_seg(
        input_path,
        "/sensing/camera/front_right/compressed_image",
        "/sensing/camera/front_right/compressed_image/gt_seg/human"
    )

    plot_img_and_seg(img, mask)

if __name__ == '__main__':
    main()