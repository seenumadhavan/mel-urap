import os
import numpy as np
import shutil

sample_list_dict = '/soge-home/projects/leaf-gpu/full_res/data/sample_list/exp1'
trn_sample = set(np.load(os.path.join(sample_list_dict, 'trn_sample.npy')))
val_sample = set(np.load(os.path.join(sample_list_dict, 'val_sample.npy')))
tst_sample = set(np.load(os.path.join(sample_list_dict, 'tst_sample.npy')))
sample_list = list(trn_sample.union(val_sample.union(tst_sample)))

for sample in sample_list:
    if '_a' in sample:
        sample_name = sample[0:-2]
        sample_list.remove(sample_name + '_a')
        sample_list.remove(sample_name + '_b')
        sample_list.append(sample_name)

result_dict = '/soge-home/projects/leaf-gpu/results'
if not os.path.isdir(result_dict):
    os.mkdir(result_dict)
for sample in sample_list:
    sample_dict = os.path.join(result_dict, sample)
    if not os.path.isdir(sample_dict):
        os.mkdir(sample_dict)

source_dict = '/soge-home/projects/leaf-gpu/full_res'
f_list = ['result']
exp_list = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6']
for f_seq in f_list:
    for exp_seq in exp_list:
        exp_dict = os.path.join(source_dict, f_seq, exp_seq)
        sample_list_s = os.listdir(exp_dict)
        for sample_s in sample_list_s:
            sample_name_s = sample_s
            if '_a' in sample_s[-2:]:
                sample_name_s = sample_s[0:-2]
            if '_b' in sample_s[-2:]:
                sample_name_s = sample_s[0:-2]
            aim_dict = os.path.join(result_dict, sample_name_s)
            aim_cont = os.listdir(aim_dict)
            if len(aim_cont) == 0:
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_img.png'), aim_dict)
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_seg.png'), aim_dict)
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_roi.png'), aim_dict)
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_mask.png'), aim_dict)
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_cnn_mask.png'), aim_dict)
                img_check = 1
                seg_check = 1
                roi_check = 1
                mask_check = 1
                cnn_mask_check = 1
            else:
                img_check = 0
                seg_check = 0
                roi_check = 0
                mask_check = 0
                cnn_mask_check = 0
                for content in aim_cont:
                    if '_img.png' in content:
                        img_check = 1
                    if '_seg.png' in content:
                        seg_check = 1
                    if '_roi.png' in content:
                        roi_check = 1
                    if '_mask.png' in content:
                        mask_check = 1
                    if '_cnn_mask.png' in content:
                        cnn_mask_check = 1
            if img_check == 0:
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_img.png'), aim_dict)
            if seg_check == 0:
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_seg.png'), aim_dict)
            if roi_check == 0:
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_roi.png'), aim_dict)
            if mask_check == 0:
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_mask.png'), aim_dict)
            if cnn_mask_check == 0:
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_cnn_mask.png'), aim_dict)
            k_cnn = 1
            for content in aim_cont:
                if '_cnn_' in content:
                    if '_mask.png' not in content:
                        k_cnn += 1
            shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_cnn.png'), aim_dict)
            os.rename(os.path.join(aim_dict, sample_s + '_cnn.png'), os.path.join(aim_dict, sample_s + '_cnn_' + str(k_cnn) + '.png'))

source_dict = '/soge-home/projects/leaf-gpu/linc3132'
f_list = ['f1', 'f2', 'f3', 'f4', 'f5']
exp_list = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6']
for f_seq in f_list:
    for exp_seq in exp_list:
        exp_dict = os.path.join(source_dict, f_seq, 'result', exp_seq)
        sample_list_s = os.listdir(exp_dict)
        for sample_s in sample_list_s:
            sample_name_s = sample_s
            if '_a' in sample_s[-2:]:
                sample_name_s = sample_s[0:-2]
            if '_b' in sample_s[-2:]:
                sample_name_s = sample_s[0:-2]
            aim_dict = os.path.join(result_dict, sample_name_s)
            aim_cont = os.listdir(aim_dict)
            if len(aim_cont) == 0:
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_img.png'), aim_dict)
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_seg.png'), aim_dict)
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_roi.png'), aim_dict)
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_mask.png'), aim_dict)
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_cnn_mask.png'), aim_dict)
                img_check = 1
                seg_check = 1
                roi_check = 1
                mask_check = 1
                cnn_mask_check = 1
            else:
                img_check = 0
                seg_check = 0
                roi_check = 0
                mask_check = 0
                cnn_mask_check = 0
                for content in aim_cont:
                    if '_img.png' in content:
                        img_check = 1
                    if '_seg.png' in content:
                        seg_check = 1
                    if '_roi.png' in content:
                        roi_check = 1
                    if '_mask.png' in content:
                        mask_check = 1
                    if '_cnn_mask.png' in content:
                        cnn_mask_check = 1
            if img_check == 0:
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_img.png'), aim_dict)
            if seg_check == 0:
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_seg.png'), aim_dict)
            if roi_check == 0:
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_roi.png'), aim_dict)
            if mask_check == 0:
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_mask.png'), aim_dict)
            if cnn_mask_check == 0:
                shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_cnn_mask.png'), aim_dict)
            k_cnn = 1
            for content in aim_cont:
                if '_cnn_' in content:
                    if '_mask.png' not in content:
                        k_cnn += 1
            shutil.copy(os.path.join(exp_dict, sample_s, sample_s + '_cnn.png'), aim_dict)
            os.rename(os.path.join(aim_dict, sample_s + '_cnn.png'), os.path.join(aim_dict, sample_s + '_cnn_' + str(k_cnn) + '.png'))
