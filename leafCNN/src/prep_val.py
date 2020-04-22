import numpy as np
import random
import os
from scipy import misc, ndimage
from PIL import Image
from time import gmtime

f_seq = 'f1'
exp_seq = 'exp1'
# scratch_dict = '/scratch'
scratch_dict = '/jewel-scratch'

sample_list_dict = os.path.join(scratch_dict + '/linc3132/' + f_seq + '/data/sample_list', exp_seq)
val_file = 'val_sample.npy'
val_sample = np.load(os.path.join(sample_list_dict, val_file))

patch_size = 256
patch_per_batch = 16
img_val = np.ndarray((val_sample.__len__() * patch_per_batch, patch_size, patch_size, 1), dtype='float32')
seg_val = np.ndarray((val_sample.__len__() * patch_per_batch, patch_size, patch_size, 1), dtype='float32')
bm_dict = scratch_dict + '/linc3132/data/Leaf_vein_binary'


for kq in range(1, 2, 1):
    sequence_name = str(kq)
    time = gmtime()
    print('No. ', sequence_name, ' starting time: ', str(time.tm_hour), ':', str(time.tm_min), ':', str(time.tm_sec))
    ks = 0
    for sample_val in val_sample:
        ks += 1
        k = 0
        img_val_sample = np.ndarray((patch_per_batch, patch_size, patch_size, 1), dtype='float32')
        seg_val_sample = np.ndarray((patch_per_batch, patch_size, patch_size, 1), dtype='float32')
        roi_file = sample_val + '_roi.png'
        roi = misc.imread(os.path.join(bm_dict, roi_file)) / 255
        img_file = sample_val + '_img.png'
        seg_file = sample_val + '_seg.png'
        img = misc.imread(os.path.join(bm_dict, img_file))
        seg = misc.imread(os.path.join(bm_dict, seg_file)) / 255
        roi[roi < 0] = 0
        roi[(roi > 0) & (roi <= 0.5)] = 0
        roi[(roi > 0.5) & (roi < 1)] = 1
        roi[roi > 1] = 1
        seg[seg < 0] = 0
        seg[(seg > 0) & (seg <= 0.5)] = 0
        seg[(seg > 0.5) & (seg < 1)] = 1
        seg[seg > 1] = 1
        seg[roi == 0] = 0

        A = len(np.nonzero(roi)[0])
        D = np.sqrt(A)
        Ct = ndimage.measurements.center_of_mass(roi)
        x1 = int(Ct[0] - D)
        x2 = int(Ct[0] + D)
        y1 = int(Ct[1] - D)
        y2 = int(Ct[1] + D)
        if x1 < 0:
            x1 = 0
        if x2 > roi.shape[0]:
            x2 = roi.shape[0]
        if y1 < 0:
            y1 = 0
        if y2 > roi.shape[1]:
            y2 = roi.shape[1]
        roi_c = roi[x1:x2, y1:y2]
        ang_choice = list(np.arange(0, 360, 1))
        ang_rotate = random.choice(ang_choice)
        roi_t = Image.fromarray(roi_c)
        roi_t = roi_t.rotate(ang_rotate)
        roi_t = np.array(roi_t)

        x = np.arange(0.75 * patch_size, roi_t.shape[0] - 0.75 * patch_size, 20)
        y = np.arange(0.75 * patch_size, roi_t.shape[1] - 0.75 * patch_size, 20)
        xx, yy = np.meshgrid(x, y, sparse=False)
        xx = np.resize(xx, (np.product(xx.shape)))
        yy = np.resize(yy, (np.product(yy.shape)))
        kk = list(range(np.product(yy.shape)))
        while k < patch_per_batch:
            kc = np.random.choice(kk)
            xc = xx[kc]
            yc = yy[kc]
            window_size = patch_size * (1 + (np.random.rand(1) - 0.5) * 0.5)
            x_min = int(xc - window_size / 2)
            x_max = int(xc + window_size / 2)
            y_min = int(yc - window_size / 2)
            y_max = int(yc + window_size / 2)
            roi_p = roi_t[x_min:x_max, y_min:y_max]

            if roi_p.min() == 1:
                img_c = img[x1:x2, y1:y2]
                seg_c = seg[x1:x2, y1:y2]

                img_t = Image.fromarray(img_c)
                img_t = img_t.rotate(ang_rotate)
                img_t = np.array(img_t)
                seg_t = Image.fromarray(seg_c)
                seg_t = seg_t.rotate(ang_rotate)
                seg_t = np.array(seg_t)

                img_p = img_t[x_min:x_max, y_min:y_max]
                seg_p = seg_t[x_min:x_max, y_min:y_max]

                img_p = misc.imresize(img_p, (patch_size, patch_size))
                seg_p = misc.imresize(seg_p, (patch_size, patch_size))
                seg_p[seg_p < 0] = 0
                seg_p[(seg_p > 0) & (seg_p <= 0.5)] = 0
                seg_p[(seg_p > 0.5) & (seg_p < 1)] = 1
                seg_p[seg_p > 1] = 1
                img_val_sample[k, ...] = img_p[np.newaxis, ..., np.newaxis]
                seg_val_sample[k, ...] = seg_p[np.newaxis, ..., np.newaxis]
                k += 1
                print(ks, '/', val_sample.__len__(), sample_val, ':', k, '/', patch_per_batch)
        img_val[(ks - 1) * patch_per_batch:ks * patch_per_batch, ...] = img_val_sample
        seg_val[(ks - 1) * patch_per_batch:ks * patch_per_batch, ...] = seg_val_sample

    time = gmtime()
    print('No. ', sequence_name, ' ending time: ', str(time.tm_hour), ':', str(time.tm_min), ':', str(time.tm_sec))

    save_dict = os.path.join(scratch_dict + '/linc3132/' + f_seq + '/data/val_data', exp_seq)
    img_val_save_name = 'img_val_' + sequence_name + '.npy'
    seg_val_save_name = 'seg_val_' + sequence_name + '.npy'
    np.save(os.path.join(save_dict, img_val_save_name), img_val)
    np.save(os.path.join(save_dict, seg_val_save_name), seg_val)
