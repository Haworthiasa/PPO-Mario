"""Ảnh động cho README từ video mà evaluate.py ghi lại (60 khung hình/giây, 256 × 240).

Lấy một khung trong mỗi `step` khung (mặc định 3, tức 20 khung hình/giây), gộp các khung liền nhau giống hệt nhau thành
một khung hiện lâu hơn, rồi ghi GIF lặp mãi.

    python experiments/video_to_gif.py submission/1-1.mp4 docs/1-1.gif
"""
import argparse

import imageio.v3 as iio
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("gif")
    ap.add_argument("--step", type=int, default=3, help="lấy một khung trong mỗi step khung của video 60 fps")
    a = ap.parse_args()
    frames, durations = [], []
    ms = round(1000 * a.step / 60)
    for i, f in enumerate(iio.imiter(a.video)):
        if i % a.step:
            continue
        if frames and np.array_equal(f, frames[-1]):
            durations[-1] += ms
        else:
            frames.append(f)
            durations.append(ms)
    imgs = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    imgs[0].save(a.gif, save_all=True, append_images=imgs[1:], duration=durations, loop=0, optimize=True)
    print(f"{a.gif}: {len(imgs)} khung, {sum(durations) / 1000:.1f} giây")


if __name__ == "__main__":
    main()
