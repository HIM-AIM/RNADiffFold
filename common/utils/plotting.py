import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import os


def get_image_grid(images, nrow=8, padding=2):
    '''
    Get a plotting-friendly grid image from images.

    Args:
        images: Tensor, shape (b, c, h, w)
        nrow: int, the number of images per row.
        padding: int, the number of padding pixels.

    Returns:
        image_grid: numpy array, shape (H,W,c), where H and W are the size of the grid image.
    '''
    image_grid = vutils.make_grid(images, nrow=nrow, padding=padding)
    image_grid = image_grid.permute([1,2,0]).detach().cpu().numpy()
    return image_grid



def plot_quantized_images(images, num_bits=8, nrow=8, padding=2):
    '''
    Plot quantized images.

    Args:
        images: Tensor, shape (b, c, h, w)
        num_bits: int, the number of bits for the quantized image.
        nrow: int, the number of images per row.
        padding: int, the number of padding pixels.
    '''
    image_grid = get_image_grid(images.float()/(2**num_bits - 1), nrow=nrow, padding=padding)
    plt.figure()
    plt.imshow(image_grid)
    plt.show()


def plot_evolution(epochs, examples, save_dir, data_name, step_name='Step', filename='evolution.gif'):
    '''Given a sequence of batches of samples, this animates their evolution.
    Useful for showing how a particular sample changes during training.
    Also useful for illustrating the reverse process.'''
    fig = plt.figure()
    im = plt.imshow(examples[0], interpolation='none')

    def init():
        fig.suptitle(data_name + '  ' + step_name + ': 0')
        im.set_data(examples[0])
        return [im]

    def animate(i):
        fig.suptitle(data_name + '  ' + step_name + ': {}'.format(epochs[i]))
        im.set_array(examples[i])
        return [im]

    # generate the animation
    ani = FuncAnimation(fig, animate, init_func=init,
                        frames=len(examples), interval=300, repeat=True)

    ani.save(os.path.join(save_dir, filename), writer='pillow', fps=100)

    fig.clf()
