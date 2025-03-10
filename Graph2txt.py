import numpy as np

tr_feat = np.load('NewVersion/new_opensmile_emolarge_featuresarrayFloat16.npy')
tr_label = np.load('NewVersion/new_opensmile_labelsarrayFloat16.npy')
print(f"Feature shape: {len(tr_feat)}")
print(f"Label shape: {len(tr_label)}")
txt = []

txt.append(tr_feat.shape[0])
for i in range(tr_feat.shape[0]):
    for j in range(tr_feat.shape[1]):
        if j == 0:
            txt.append('%s %s' % (tr_feat.shape[1], int(tr_label[i])))
        
       
        value = tr_feat[i, j].astype('float16')  
        
        if j == tr_feat.shape[1] - 1:
            txt.append('%s 1 %s %s' % (j, j - 1, value))
        else:
            txt.append('%s 1 %s %s' % (j, j + 1, value))

np.savetxt('shemoGraph.txt', txt, fmt='%s')

# import numpy as np

# tr_feat = np.load('NewVersion/new_opensmile_emolarge_featuresarray.npy')
# tr_label = np.load('NewVersion/new_opensmile_labelsarray.npy')
# print(f"Feature shape: {tr_feat.shape}")
# print(f"Label shape: {tr_label.shape}")

# tr_feat = tr_feat / np.max(np.abs(tr_feat), axis=0, keepdims=True)

# tr_feat = tr_feat.astype('float16')

# txt = []
# txt.append(tr_feat.shape[0])


# for i in range(tr_feat.shape[0]):
#     for j in range(tr_feat.shape[1]):
#         if j == 0:
#             txt.append('%s %s' % (tr_feat.shape[1], int(tr_label[i])))

#         value = tr_feat[i, j]  
#         if j == tr_feat.shape[1] - 1:
#             txt.append('%s 1 %s %s' % (j, j - 1, value))
#         else:
#             txt.append('%s 1 %s %s' % (j, j + 1, value))


# np.savetxt('shemoGraph.txt', txt, fmt='%s')

