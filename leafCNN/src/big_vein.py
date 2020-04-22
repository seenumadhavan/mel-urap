import os
import shutil

aim_dict = '/soge-home/projects/leaf-gpu/results'
aim_c = os.listdir(aim_dict)

big_dict = '/soge-home/projects/leaf-gpu/linc3132/data/Big_vein_binary'
big_c = os.listdir(big_dict)
for file in big_c:
    file_name = file[0:-13]
    print(file_name)
    for folder in aim_c:
        if file_name in folder:
            folder_dict = os.path.join(aim_dict, folder)
            print(folder_dict)
            shutil.copy(os.path.join(big_dict, file), folder_dict)

'''
for folder in aim_c:
    folder_dict = os.path.join(aim_dict, folder)
    folder_c = os.listdir(folder_dict)
    for file in folder_c:
        if '_big_mask.png' in file:
            print(file)
            os.remove(os.path.join(folder_dict, file))
'''