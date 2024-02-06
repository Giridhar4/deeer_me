import numpy as np
import cv2 as cv2
import scipy.io
import os
from pystackreg import StackReg
from PIL import Image

thermal_image = '/Users/giridharpeddi/OnChip/Compilations/pythonProject2/Data/test_thermal2.png'
rgb_image = '/Users/giridharpeddi/OnChip/Compilations/pythonProject2/Data/test_rgb2.png'
nshifts = 24
scale = 4
shift_max = 10
see = True
thermal = False

def conv(image):
    im_numpy = cv2.imread(image)
    im_numpy = np.copy(im_numpy, order='C', ).astype(np.float32)

    print(im_numpy.shape)
    return np.copy(im_numpy[..., 1], order='C').astype(np.float32)


def translation_matrix(shift):
    mat = np.array([[1, 0, shift[0]],
                    [0, 1, shift[1]]])
    return mat


def stack_jitter(im, im_t):
    H, W = im.shape
    Ht, Wt = im_t.shape
    shifts = -shift_max + 2 * shift_max * np.random.rand(nshifts, 2)
    Y, X = np.mgrid[:H, :W]
    Yt, Xt = np.mgrid[:Ht, :Wt]
    tmp = cv2.resize(im, None, fx=1 / scale, fy=1 / scale)
    tmp2 = cv2.resize(im_t, None, fx=1 / scale, fy=1 / scale)
    Hl, Wl = tmp.shape
    Htl, Wtl = tmp2.shape
    imstack = np.zeros((nshifts, Hl, Wl), dtype=np.float32)
    imstack_t = np.zeros((nshifts, Htl, Wtl), dtype=np.float32)
    mats = np.zeros((nshifts, 2, 3))
    t_mats = np.zeros((nshifts, 2, 3))
    # Ensure first shift and theta are zero
    shifts[0, :] = 0
    coords = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), np.ones((H * W, 1))))
    t_coords = np.hstack((Xt.reshape(-1, 1), Yt.reshape(-1, 1), np.ones((Ht * Wt, 1))))

    for idx in range(nshifts):
        shift = shifts[idx, :]
        mat = translation_matrix(shift)
        mats[idx, ...] = mat
        t_mats[idx, ...] = mat
        coords_new = mat.dot(coords.T).T
        t_coords_new = mat.dot(t_coords.T).T

        Xnew = coords_new[:, 0].reshape(H, W)
        Ynew = coords_new[:, 1].reshape(H, W)
        Xtnew = t_coords_new[:, 0].reshape(Ht, Wt)
        Ytnew = t_coords_new[:, 1].reshape(Ht, Wt)

        Xnew = cv2.resize(Xnew, (Wl, Hl), interpolation=cv2.INTER_LINEAR)
        Ynew = cv2.resize(Ynew, (Wl, Hl), interpolation=cv2.INTER_LINEAR)
        Xtnew = cv2.resize(Xtnew, (Wtl, Htl), interpolation=cv2.INTER_LINEAR)
        Ytnew = cv2.resize(Ytnew, (Wtl, Htl), interpolation=cv2.INTER_LINEAR)

        imstack[idx, ...] = cv2.remap(im, Xnew.astype(np.float32),
                                      Ynew.astype(np.float32),
                                      cv2.INTER_LINEAR)
        imstack_t[idx, ...] = cv2.remap(im_t, Xtnew.astype(np.float32),
                                        Ytnew.astype(np.float32),
                                        cv2.INTER_LINEAR)
    return imstack, imstack_t, coords, t_coords, mats


def register_stack(imstack, full_res, method=StackReg.TRANSLATION):
    nimg, h, w = imstack.shape
    hr, wr = full_res
    imstack_full = np.zeros((nimg, hr, wr))
    for idx in range(nimg):
        imstack_full[idx, ...] = cv2.resize(imstack[idx, ...], (wr, hr),
                                            interpolation=cv2.INTER_AREA)
    reg = StackReg(method)
    reg_mats = reg.register_stack(imstack_full, reference='first', verbose=True)
    return reg_mats


imt = conv(thermal_image)
im = conv(rgb_image)
rgbstack, tstack, coord, t_coord, mats = stack_jitter(im, imt)
_, hl, wl = rgbstack.shape
_, htl , wtl = tstack.shape
ecc_mats = register_stack(rgbstack, (hl, wl))
ecc_t_mats = register_stack(tstack, (htl, wtl))
mdict = {'rgb': rgbstack,
         'thermal': tstack,
         'ecc_mat': ecc_mats,
         'ecc_t_mat': ecc_t_mats,
         'cordinates': coord,
         'mats': mats}

scipy.io.savemat('current_matfile.mat',mdict)
if see:
    mat_file = scipy.io.loadmat('/Users/giridharpeddi/OnChip/Compilations/pythonProject2/current_matfile.mat')
    if thermal:
        image_data = mat_file['thermal']
        output_folder = '/Users/giridharpeddi/OnChip/Compilations/pythonProject2/results/thermal_images'
        os.makedirs(output_folder, exist_ok=True)
    else:
        image_data = mat_file['rgb']
        output_folder = '/Users/giridharpeddi/OnChip/Compilations/pythonProject2/results/rgb_images'
        os.makedirs(output_folder, exist_ok=True)
    for i, image in enumerate(image_data):
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        image_filename = os.path.join(output_folder, f'image_{i:03d}.png')
        pil_image.save(image_filename)
    print(f"{len(image_data)} images saved as PNGs in {output_folder}")
    image_folder = output_folder
    output_video_path = '/Users/giridharpeddi/OnChip/Compilations/pythonProject2/results/thermal3.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()