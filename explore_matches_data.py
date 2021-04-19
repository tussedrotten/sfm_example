import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from dataset import read_dataset


def visualize_covisibility(matched_frames):
    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return

        i = int(event.ydata)
        j = int(event.xdata)
        visualize_matches(matched_frames[i], matched_frames[j])

    num_frames = len(matched_frames)

    covisibility_mat = np.zeros([num_frames, num_frames])

    for i in range(num_frames):
        for j in range(i, num_frames):
            covisibility_mat[i, j] = matched_frames[i].number_of_matches(matched_frames[j])
            covisibility_mat[j, i] = matched_frames[i].number_of_matches(matched_frames[j])

    # Use Qt 5 backend in visualisation.
    matplotlib.use('qt5agg')

    # Create axis.
    fig = plt.figure()
    ax = plt.axes()
    artist = ax.imshow(covisibility_mat)
    fig.colorbar(artist, ax=ax)
    fig.canvas.mpl_connect('button_press_event', on_click)
    ax.set_title("Number of matches between frames.\nPress an element to show matches")
    plt.show()


def visualize_matches(frame0, frame1):
    kp0, _, kp1, _ = frame0.extract_correspondences_with_frame(frame1)
    draw_matches(frame0.load_image(), kp0, frame1.load_image(), kp1)


def draw_matches(img1, kp1, img2, kp2):
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    else:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])

    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

    # Create matches axis.
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0.1, hspace=0, wspace=0)
    ax.imshow(new_img)

    match_artists = []
    for i in range(kp1.shape[1]):
        match_line = np.c_[kp1[:, i], kp2[:, i] + np.array([img1.shape[1], 0])]
        match_artists += ax.plot(match_line[0, :], match_line[1, :], '-o', fillstyle='none')

    ax.set_axis_off()

    # Add slider.
    widget_color = 'lightgoldenrodyellow'
    slider = Slider(plt.axes([0.2, 0.05, 0.7, 0.03], facecolor=widget_color),
                    'Sample every ...', 1, 100, valinit=1, valstep=5)

    def update_matches(val=None):
        n = int(slider.val)
        for i, artist in enumerate(match_artists):
            artist.set_visible((i % n) == 0)

    slider.on_changed(update_matches)
    fig.show()


def main():
    matched_frames, _ = read_dataset()
    visualize_covisibility(matched_frames)


if __name__ == '__main__':
    main()
