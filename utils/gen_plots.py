# save the video

import os
from PIL import Image
import imageio


def generate_gif(test_folder, count, remove_imgs=True):
    n = count  # The number of pictures to be stitched into a video
    fps = n // 5  # The frame rate of the generated video
    imgs = []
    for i in range(1, n, 1):
        img_file = f"{test_folder}/approx_{i}.png"
        img = imageio.imread(img_file)
        img = Image.fromarray(img)  # .resize((size_width, size_height))
        imgs.append(img)
        os.remove(img_file) if remove_imgs else None
    filename1 = os.path.join(test_folder, "0_video_u.mp4")
    filename2 = os.path.join(test_folder, "0_video_u.gif")
    imageio.mimwrite(filename1, imgs, fps=fps)  #
    imageio.mimsave(filename2, imgs, format='GIF')

    imgs = []
    for i in range(1, n, 1):
        img_file = f"{test_folder}/contour_approx_{i}.png"
        img = imageio.imread(img_file)
        img = Image.fromarray(img)  # .resize((size_width, size_height))
        imgs.append(img)
        os.remove(img_file) if remove_imgs else None
    filename = os.path.join(test_folder, "0_video_u_contour.mp4")
    imageio.mimwrite(filename, imgs, fps=fps)
