import json 
import os
import numpy as np


datadir_old="/home/michael/Projects/test_shapeNet/ShapeNet/ShapeNetCore.v2"

f=open(os.path.join(datadir_old,'split_original','mugs_split.json'))
dict=json.load(f)['id_test_list']
print(len(dict))

f=open(os.path.join(datadir_old,'split_original','mugs_split.json'))
dict_train=json.load(f)['id_train_list']

#print(dict_train)

test_id=[]
    
for id in dict:
    #print(id[1])
    try:
        np.load('/home/michael/Projects/test_shapeNet/ShapeNet/ShapeNetCore.v2/03797390/{}/models/model_b_0_uniform_occ.npz'.format(id[1]))
        #test_id.append(id)
    except FileNotFoundError:
        pass
    else:
        test_id.append(id)


d={'id_train_list':dict_train,'id_test_list':test_id}

print(len(d['id_test_list']))

with open(os.path.join(datadir_old,'split_original','mugs_split.json'), 'w') as fp:
    json.dump(d, fp)
