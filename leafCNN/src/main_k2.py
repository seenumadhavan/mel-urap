from cnn import prep_val, train, val_dice, test

'''
# validation data prep
prep_val('f1', 'exp5', '/jewel-scratch')
prep_val('f2', 'exp5', '/jewel-scratch')
prep_val('f3', 'exp5', '/jewel-scratch')
prep_val('f4', 'exp5', '/jewel-scratch')
prep_val('f5', 'exp5', '/jewel-scratch')
'''

# training
# train('f1', 'exp5', '/jewel-scratch', 23, 39)
# train('f2', 'exp5', '/jewel-scratch', 33, 39)
# train('f3', 'exp5', '/jewel-scratch', 31, 39)
# train('f4', 'exp5', '/soge-home/projects/leaf-gpu', 34, 39)
# train('f5', 'exp5', '/soge-home/projects/leaf-gpu', 32, 39)

# validating
# val_dice('f1', 'exp5', '/jewel-scratch')
# val_dice('f2', 'exp5', '/jewel-scratch')
# val_dice('f3', 'exp5', '/soge-home/projects/leaf-gpu')
# val_dice('f4', 'exp5', '/soge-home/projects/leaf-gpu')
# val_dice('f5', 'exp5', '/soge-home/projects/leaf-gpu')

# testing
# test('f1', 'exp5', '/jewel-scratch', 0)
# test('f2', 'exp5', '/soge-home/projects/leaf-gpu', 17)
# test('f3', 'exp5', '/soge-home/projects/leaf-gpu', 0)
# test('f4', 'exp5', '/soge-home/projects/leaf-gpu', 0)
# test('f5', 'exp5', '/soge-home/projects/leaf-gpu', 0)
test('f6', 'exp2', '/soge-home/projects/leaf-gpu', 50)
