import numpy as np
import random
import os
import imageio
from scipy import misc, ndimage
from PIL import Image
from time import gmtime
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization
from keras.optimizers import Adam
from keras import losses
import h5py

def dice(bm_pred, bm_val):
    intersection = np.multiply(bm_pred, bm_val)
    seg_count = np.sum(bm_pred)
    ref_count = np.sum(bm_val)
    int_count = np.sum(intersection)
    dice_similarity = 2 * int_count / (ref_count + seg_count)
    return dice_similarity


def prep_val():
    #assuming this script is run from the badLeafCNN folder
    #sample_list_dict = os.getcwd()
    #val_file = 'val_sample.npy'
    #val_sample = np.load(os.path.join(sample_list_dict, val_file))
    val_sample = ['test']
    print('hello')

    patch_size = 256
    patch_per_batch = 16
    img_val = np.ndarray((val_sample.__len__() * patch_per_batch, patch_size, patch_size, 1), dtype='float32')
    seg_val = np.ndarray((val_sample.__len__() * patch_per_batch, patch_size, patch_size, 1), dtype='float32')
    bm_dict = os.getcwd()
    #Clahe'd stuff

    sequence_name = str(1)
    time = gmtime()
    print('No. ', sequence_name, ' starting time: ', str(time.tm_hour), ':', str(time.tm_min), ':', str(time.tm_sec))
    ks = 0
    for sample_val in val_sample:
        ks += 1
        k = 0
        img_val_sample = np.ndarray((patch_per_batch, patch_size, patch_size, 1), dtype='float32')
        #each batch is an array of patches from a single image
        seg_val_sample = np.ndarray((patch_per_batch, patch_size, patch_size, 1), dtype='float32')
        roi_file = sample_val + '_roi.png'
        #yellow
        #now each pixel is between 0 and 1
        roi = Image.open(os.path.join(bm_dict, roi_file)).convert("RGB")
        roi = np.asarray(roi) / 255
        img_file = sample_val + '_img.png'
        #plain image
        #file with segmentation drawn
        seg_file = sample_val + '_seg.png'
        img = Image.open(os.path.join(bm_dict, img_file)).convert("RGB")
        img = np.asarray(img)
        #now each pixel is between 0 and 1
        seg = Image.open(os.path.join(bm_dict, seg_file)).convert("RGB")
        seg = np.asarray(seg) / 255

        #making each image binary
        roi[roi < 0] = 0
        roi[(roi > 0) & (roi <= 0.5)] = 0
        roi[(roi > 0.5) & (roi < 1)] = 1
        roi[roi > 1] = 1
        seg[seg < 0] = 0
        seg[(seg > 0) & (seg <= 0.5)] = 0
        seg[(seg > 0.5) & (seg < 1)] = 1
        seg[seg > 1] = 1
        seg[roi == 0] = 0
        print(roi.shape)
        print(seg.shape)
        roi_img = Image.fromarray(roi, 'RGB')
        seg_img = Image.fromarray(seg, 'RGB')
        img_img = Image.fromarray(img, 'RGB')
        roi_img.save('roi_img.png')
        seg_img.save('seg_img.png')
        img_img.save('img_img.png')
        roi_img.show()
        seg_img.show()
        img_img.show()

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
    return


def train(f_seq, exp_seq, scratch_dict, init_val, end_val):
    sample_list_dict = os.path.join(scratch_dict, 'linc3132/' + f_seq + '/data/sample_list', exp_seq)
    trn_file = 'trn_sample.npy'
    #file names for only training
    trn_sample = np.load(os.path.join(sample_list_dict, trn_file))

    patch_size = 256
    patch_per_batch = 32
    data_folder = os.path.join(scratch_dict, 'linc3132/data/Leaf_vein_augment')

    val_dict = os.path.join(scratch_dict, 'linc3132/' + f_seq + '/data/val_data')
    for k_init in range(init_val, end_val, 1):
        print('Epoch:', str(k_init + 1))
        k_init += 1
        img_trn = np.ndarray((trn_sample.__len__() * patch_per_batch, patch_size, patch_size, 1), dtype='float32')
        seg_trn = np.ndarray((trn_sample.__len__() * patch_per_batch, patch_size, patch_size, 1), dtype='float32')
        ks = 0
        for sample_train in trn_sample:
            ks += 1
            sample_folder = os.path.join(data_folder, sample_train)
            folder_content = os.listdir(sample_folder)
            img_list = [x for x in folder_content if 'img' in x]
            img_sample = img_list[k_init]
            seg_sample = 'seg' + img_sample[3:]
            img_trn_sample = np.load(os.path.join(sample_folder, img_sample))
            seg_trn_sample = np.load(os.path.join(sample_folder, seg_sample))
            img_trn[(ks - 1) * patch_per_batch:ks * patch_per_batch, ...] = img_trn_sample
            seg_trn[(ks - 1) * patch_per_batch:ks * patch_per_batch, ...] = seg_trn_sample
        img_val = np.load(os.path.join(val_dict, exp_seq, 'img_val_1.npy'))
        seg_val = np.load(os.path.join(val_dict, exp_seq, 'seg_val_1.npy'))
        img_train = np.concatenate((img_trn, img_val), axis=0)
        seg_train = np.concatenate((seg_trn, seg_val), axis=0)
        val_split = img_val.__len__() / img_train.__len__()

        model_folder = os.path.join(scratch_dict, 'linc3132/' + f_seq + '/model', exp_seq)
        model_list = os.listdir(model_folder)
        if model_list.__len__() == 0:
            patch_rows = 256
            patch_cols = 256
            #added he_normal initialization
            inputs = Input((patch_rows, patch_cols, 1))
            conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            conv1 = BatchNormalization()(conv1)
            conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            conv1 = BatchNormalization()(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = BatchNormalization()(conv2)
            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            conv2 = BatchNormalization()(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = BatchNormalization()(conv3)
            conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            conv3 = BatchNormalization()(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = BatchNormalization()(conv4)
            conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            conv4 = BatchNormalization()(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

            conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
            conv5 = BatchNormalization()(conv5)
            conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            conv5 = BatchNormalization()(conv5)

            up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
            conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
            conv6 = BatchNormalization()(conv6)
            conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
            conv6 = BatchNormalization()(conv6)

            up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
            conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
            conv7 = BatchNormalization()(conv7)
            conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
            conv7 = BatchNormalization()(conv7)

            up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
            conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
            conv8 = BatchNormalization()(conv8)
            conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
            conv8 = BatchNormalization()(conv8)

            up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
            conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up9)
            conv9 = BatchNormalization()(conv9)
            conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = BatchNormalization()(conv9)

            conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

            model = Model(inputs=[inputs], outputs=[conv10])

            epoch_number = 0

        else:
            model_file = model_list[-1]
            model_idx_list = []
            for model_sample in model_list:
                model_idx = int(model_sample[8:10])
                model_idx_list.append(model_idx)
            epoch_number = int(np.max(model_idx_list))
            if epoch_number < 10:
                model_file_ref = model_file[0:8] + '0' + str(epoch_number) + model_file[10:11]
            else:
                model_file_ref = model_file[0:8] + str(epoch_number) + model_file[10:11]
            for model_sample in model_list:
                if model_file_ref in model_sample:
                    model_file = model_sample
            f_model = h5py.File(os.path.join(model_folder, model_file), 'r+')
            if 'optimizer_weights' in f_model:
                del f_model['optimizer_weights']
            f_model.close()
            model = load_model(os.path.join(model_folder, model_file))

        model.compile(optimizer=Adam(lr=1e-4), loss=losses.binary_crossentropy, metrics = ['accuracy'])

        if k_init < 10:
            file_name = 'weights.' + '0' + str(k_init) + '-{val_loss:.3f}.h5'
        else:
            file_name = 'weights.' + str(k_init) + '-{val_loss:.3f}.h5'
        model_check_file = os.path.join(model_folder, file_name)

        model_checkpoint = ModelCheckpoint(model_check_file, monitor='val_loss', save_best_only=False)

        model.fit(img_train, seg_train, batch_size=16, epochs=1, verbose=1, shuffle=True, validation_split=val_split,
                  callbacks=[model_checkpoint])
    return

def val_dice(f_seq, exp_seq, scratch_dict):
    file_dict = os.path.join(scratch_dict, 'linc3132/' + f_seq + '/data/val_data', exp_seq)
    img_val_set = np.load(os.path.join(file_dict, 'img_val_1.npy'))
    seg_val_set = np.load(os.path.join(file_dict, 'seg_val_1.npy'))

    model_folder = os.path.join(scratch_dict, 'linc3132/' + f_seq + '/model', exp_seq)
    model_list_init = os.listdir(model_folder)
    epoch_number = 0
    model_file_init = model_list_init[0]
    model_list = []
    for model_num in range(len(model_list_init)):
        epoch_number += 1
        if epoch_number < 10:
            str_cmp = str(model_file_init[0:8] + '0' + str(epoch_number))
            model_file = [s for s in model_list_init if str_cmp in s]
        else:
            str_cmp = str(model_file_init[0:8] + str(epoch_number))
            model_file = [s for s in model_list_init if str_cmp in s]
        model_list.append(str(model_file[0]))

    D = []
    km = 0
    for model_file in model_list:
        km += 1
        print('Progress: ', km, '/', len(model_list))
        f_model = h5py.File(os.path.join(model_folder, model_file), 'r+')
        if 'optimizer_weights' in f_model:
            del f_model['optimizer_weights']
        f_model.close()
        model = load_model(os.path.join(model_folder, model_file))

        prd_val_set = model.predict(img_val_set, batch_size=16)
        prd_val_set[prd_val_set >= 0.5] = 1
        prd_val_set[prd_val_set < 0.5] = 0
        d = dice(prd_val_set, seg_val_set)
        D.append(d)
    save_dict = os.path.join(scratch_dict, 'linc3132/' + f_seq + '/log', exp_seq)
    np.save(os.path.join(save_dict, 'dice_val.npy'), D)
    return


def test(f_seq, exp_seq, scratch_dict, kt):
    file_dict = os.path.join(scratch_dict, 'linc3132/data/Leaf_vein_binary')
    sample_list_dict = os.path.join(scratch_dict, 'linc3132/' + f_seq + '/data/sample_list', exp_seq)
    tst_file = 'tst_sample.npy'
    tst_sample = np.load(os.path.join(sample_list_dict, tst_file))

    model_folder = os.path.join(scratch_dict, 'linc3132/' + f_seq + '/model', exp_seq)
    D_list_dict = os.path.join(scratch_dict, 'linc3132/' + f_seq + '/log', exp_seq)
    D_list = np.load(os.path.join(D_list_dict, 'dice_val.npy'))
    D_idx = np.argmax(D_list)
    model_list = os.listdir(model_folder)
    model_file = model_list[D_idx]
    f_model = h5py.File(os.path.join(model_folder, model_file), 'r+')
    if 'optimizer_weights' in f_model:
        del f_model['optimizer_weights']
    f_model.close()
    model = load_model(os.path.join(model_folder, model_file))

    res_save_dict = os.path.join(scratch_dict, 'linc3132/' + f_seq + '/result', exp_seq)
    for tst_case in tst_sample[kt:]:
        kt += 1
        print('Test case: ' + tst_case + 'Progress: ' + str(kt) + ' / ' + str(len(tst_sample)))
        if not os.path.exists(os.path.join(res_save_dict, tst_case)):
            os.mkdir(os.path.join(res_save_dict, tst_case))
        img_tst_sample = tst_case + '_img.png'
        msk_tst_sample = tst_case + '_slc.png'
        roi_tst_sample = tst_case + '_roi.png'
        seg_tst_sample = tst_case + '_seg.png'
        img = misc.imread(os.path.join(file_dict, img_tst_sample))
        msk = misc.imread(os.path.join(file_dict, msk_tst_sample))
        roi = misc.imread(os.path.join(file_dict, roi_tst_sample))
        seg = misc.imread(os.path.join(file_dict, seg_tst_sample))

        patch_size = 256
        x = np.arange(0, img.shape[0] - patch_size, 50)
        y = np.arange(0, img.shape[1] - patch_size, 50)
        xx, yy = np.meshgrid(x, y, sparse=True)
        pred_mask = np.zeros(img.shape)
        pred_mask[0:xx[0][-1], 0:yy[-1][0]] = 1

        sample = np.ndarray((len(xx[0, ...]) * len(yy[..., 0]), patch_size, patch_size, 1), dtype='float32')
        k = 0
        for xi in xx[0, ...]:
            for yi in yy[..., 0]:
                x_min = int(xi)
                x_max = int(xi + patch_size)
                y_min = int(yi)
                y_max = int(yi + patch_size)
                sample[k, ..., 0] = img[x_min:x_max, y_min:y_max]
                k += 1

        pred_sample = model.predict(sample, batch_size=1)
        pred_seg = np.zeros(img.shape)
        weit_seg = np.zeros(img.shape)
        k = 0
        for xi in xx[0, ...]:
            for yi in yy[..., 0]:
                x_min = int(xi)
                x_max = int(xi + patch_size)
                y_min = int(yi)
                y_max = int(yi + patch_size)
                pred_seg[x_min:x_max, y_min:y_max] += pred_sample[k, ..., 0]
                weit_seg[x_min:x_max, y_min:y_max] += np.ones((patch_size, patch_size))
                k += 1
        weit_seg[weit_seg > 0] = 1 / weit_seg[weit_seg > 0]
        pred_res = np.multiply(pred_seg, weit_seg)
        misc.imsave(os.path.join(res_save_dict, tst_case, tst_case + '_cnn.png'), pred_res)
        misc.imsave(os.path.join(res_save_dict, tst_case, tst_case + '_mask.png'), msk)
        misc.imsave(os.path.join(res_save_dict, tst_case, tst_case + '_roi.png'), roi)
        misc.imsave(os.path.join(res_save_dict, tst_case, tst_case + '_seg.png'), seg)
        misc.imsave(os.path.join(res_save_dict, tst_case, tst_case + '_img.png'), img)
        misc.imsave(os.path.join(res_save_dict, tst_case, tst_case + '_cnn_mask.png'), pred_mask)
    return

prep_val()



