import numpy as np
import json 
import os
import shutil
import torch
import random
import logging
import math


import core
import multiprocessing




import argparse
import json
import logging
import os
import random
import math
import multiprocessing

#import torch
import numpy as np
from collections import defaultdict

import core
import igl


from build import main as build_pickle
from train import main_function as train
from reconstruct_2 import reconstruct as reconstruct

datadir_1="/home/michael/Projects/test_shapeNet/ShapeNet/ShapeNetCore.v2"
datadir_2="/home/michael/Projects/test_shapeNet_inference2/ShapeNet/ShapeNetCore.v2"

path_deepmend=os.getcwd()
train_test_file='/home/michael/Projects/test_shapeNet/ShapeNet/ShapeNetCore.v2/split_original/mugs_split.json'

f=open(train_test_file)
test_dict=json.load(f)['id_test_list']

experiment_dir=os.path.join('experiments','mugs')

##### create 2 dictionary, depending if first inference step was successful
if False:
    L=[]
    M=[]
    for k,id in enumerate(test_dict):
        try:
            p='/home/michael/Projects/DeepMend_changes3_new2/deepmend/experiments/mugs/Reconstructions/deepmend@b/Meshes'
            igl.read_triangle_mesh(os.path.join(p,'{}_2_.obj'.format(k)))
            L.append(id)
        
        except ValueError:
            M.append(id)
            pass
    json_dict='/home/michael/Projects/test_shapeNet/ShapeNet/ShapeNetCore.v2/split_original/mugs_split_firststep_correct.json'
    d={"id_train_list": [],"id_test_list": L}
    with open(json_dict, 'w') as fp:
        json.dump(d, fp)

    json_dict='/home/michael/Projects/test_shapeNet/ShapeNet/ShapeNetCore.v2/split_original/mugs_split_firststep_false.json'
    d={"id_train_list": [],"id_test_list": M}
    with open(json_dict, 'w') as fp:
        json.dump(d, fp)
print(len(test_dict))
for k,id in enumerate(test_dict[:]):
    print((k,id))
    #if k==7 or k==5 or k==0:
    #    continue
    ##### create dictionary/json file and save it in corresponding file
    json_dict='/home/michael/Projects/test_shapeNet/ShapeNet/ShapeNetCore.v2/mugs_split_single.json'
    d={"id_train_list": [id],"id_test_list": [id]}
    with open(json_dict, 'w') as fp:
        json.dump(d, fp)

    #####create pickle file 

    outfile_train=os.path.join(datadir_1,'mugs_train_single.pkl')
    outfile_test=os.path.join(datadir_1,'mugs_test_single.pkl')
    build_pickle(datadir_1,
                 outfile_train,
                 None,
                 None,
                 outfile_test,
                 json_dict,
                 1,
                 True,
                 True,
                 False,
                 False,
                 False,
                 skip_ask_overwirite=True)
    
    outfile_train_2=os.path.join(datadir_2,'mugs_train_single.pkl')
    outfile_test_2=os.path.join(datadir_2,'mugs_test_single.pkl')

    build_pickle(datadir_2,
                 outfile_train_2,
                 None,
                 None,
                 outfile_test_2,
                 json_dict,
                 1,
                 True,
                 True,
                 False,
                 False,
                 False,
                 skip_ask_overwirite=True)
    
    
    ###### Load model and latent codes
    model=torch.load('/home/michael/Projects/DeepMend_changes3/deepmend/experiments/mugs/ModelParameters/latest.pth')
    torch.save(model,os.path.join(path_deepmend,'experiments','mugs','ModelParameters/begin.pth'))

    shutil.copy(os.path.join(path_deepmend,'experiments','mugs','LatentCodesInference1','{}.pth'.format(k)),os.path.join(path_deepmend,'experiments','mugs','LatentCodes','begin.pth'))
    shutil.copy(os.path.join(path_deepmend,'experiments','mugs','LatentCodesInference1','{}_tool.pth'.format(k)),os.path.join(path_deepmend,'experiments','mugs','LatentCodes','begin_tool.pth'))


    
    


    ######Train model with all weights
    train_source_path=os.path.join(datadir_2,'mugs_train_single.pkl')

    train(experiment_dir,
          'begin',
          1,
          train_split_path=train_source_path,
          data_source_path=datadir_2,
          save_model_end_index=k)

    #####load model weights in right file for inference/shape creation

    #load model weights in modelparameters and save under 'inference'
    shutil.copy(os.path.join(path_deepmend,'experiments','mugs','ModelInference','{}_model.pth'.format(k)),os.path.join(path_deepmend,'experiments','mugs','ModelParameters','inference.pth'))

    #load latent codes directly via optional parameter in function
    latent_path=os.path.join(path_deepmend,'experiments','mugs','ModelInference','{}_lat.pth'.format(k))
    tool_latent_path=os.path.join(path_deepmend,'experiments','mugs','ModelInference','{}_tool_lat.pth'.format(k))

    
    #####inference for one object

    #specs_filename = core.find_specs(experiment_dir)
    specs = json.load(open(os.path.join(path_deepmend,experiment_dir,'specs.json')))
    #args.experiment_directory = os.path.dirname(specs_filename)

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    lambda_ner = 0
    lambda_prox = 0
    

    num_samples = 1000
    num_iterations = 0
    code_reg_lambda = 0
    lr = 0
    name = "ours_"
    render_threads = 4
    overwrite_codes = True
    overwrite_meshes = True
    overwrite_evals = False
    overwrite_renders = False
    threads = 2
    save_iter = False
    uniform_ratio = 0.2
    mesh_only = False


    
    test_split_file = specs["TestSplit"]
    break_latent_size = specs["BreakCodeLength"]

    network_outputs = (0, 1, 2, 3)
    total_outputs = (0, 1, 2, 3)
    composite = [(1, 2), (1, 3)]
    render_resolution = (200, 200)
    do_code_regularization = True
    isosurface_level = 0.5
    use_sigmoid = True

    

    network_kwargs = dict(
        decoder_kwargs=dict(
            latent_size=latent_size,
            tool_latent_size=break_latent_size,
            num_dims=3,
            do_code_regularization=do_code_regularization,
            **specs["NetworkSpecs"],
            **specs["SubnetSpecs"],
        ),
        decoder_constructor=arch.Decoder,
        experiment_directory=experiment_dir, #args.experiment_directory,
        checkpoint='inference',#args.checkpoint,
    )
    
    reconstruction_kwargs = dict(
        num_iterations=num_iterations,
        latent_size=latent_size,
        break_latent_size=break_latent_size,
        lambda_ner=lambda_ner,
        lambda_prox=lambda_prox,
        stat=0.01,  # [emp_mean,emp_var],
        clamp_dist=None,
        num_samples=num_samples,
        lr=lr,
        l2reg=do_code_regularization,
        code_reg_lambda=code_reg_lambda,
        loss_version=None,
        path_latent=latent_path,
        path_tool_latent=tool_latent_path
    )
    
    
    mesh_kwargs = dict(
        dims=[256, 256, 256],
        level=isosurface_level,
        gradient_direction="descent",
        batch_size=2 ** 14,
    )

    
    # Create and load the dataset
    reconstruction_handler = core.handler.ReconstructionHandler(
        experiment_directory=experiment_dir,# args.experiment_directory,
        dims=[256, 256, 256],
        name=name,
        checkpoint='inference',
        overwrite=False,
        use_occ=specs["UseOccupancy"],
        signiture=[],
    )
    sdf_dataset = core.data.SamplesDataset(
        os.path.join(datadir_1,'mugs_test_single.pkl'), #test_split_file,
        subsample=num_samples,
        uniform_ratio=uniform_ratio,
        use_occ=specs["UseOccupancy"],
        root=datadir_1,
    )
    
    reconstruct_list = list(range(len(sdf_dataset)))
   

    input_list, path_list = [], []
    if not mesh_only:
        for ii in reconstruct_list:

            # Generate the code if necessary
            path_code = reconstruction_handler.path_code(ii, create=True)
            if (not os.path.exists(path_code)) or overwrite_codes:
                if save_iter:
                    input_list.append(
                        dict(
                            test_sdf=sdf_dataset.get_broken_sample(ii),
                            iter_path=reconstruction_handler.path_values(
                                ii, 1, create=True
                            ),
                        )
                    )
                else:
                    input_list.append(
                        dict(
                            test_sdf=sdf_dataset.get_broken_sample(ii),
                        )
                    )
                path_list.append(path_code)
    

    # Spawn a threadpool to do reconstruction
    num_tasks = len(input_list)
    STATUS_INDICATOR = core.utils_multiprocessing.MultiprocessBar(num_tasks)

    def callback():
        global STATUS_INDICATOR
        global STATUS_COUNTER
        try:
            STATUS_INDICATOR.increment()
        except AttributeError:
            print("Completed: {}".format(STATUS_COUNTER))
            STATUS_COUNTER += 1
    
    
    
    if True:    
        core.utils_multiprocessing.reconstruct_chunk(input_list,
                                    path_list,
                                    reconstruct,
                                    network_kwargs,
                                    reconstruction_kwargs,
                                    overwrite_codes,
                                    callback
                                    )
        
        input_list, path_list = [], []
        for ii in reconstruct_list:


            # Generate the mesh if necessary
            for shape_idx in network_outputs:
                path_mesh = reconstruction_handler.path_mesh(ii, shape_idx, create=True)
                path_values = reconstruction_handler.path_values(ii, shape_idx)
                if (
                    not os.path.exists(path_mesh)
                    or not os.path.exists(path_values)
                    or overwrite_meshes
                ):
                    sigmoid = True
                    if shape_idx in [1, 2]:
                        sigmoid = False
                    if os.path.exists(reconstruction_handler.path_code(ii)):
                        input_list.append(
                            dict(
                                vec=reconstruction_handler.get_code(ii),
                                use_net=shape_idx,
                                save_values=path_values,
                                sigmoid=sigmoid,
                            )
                        )
                        path_list.append(path_mesh)
        core.utils_multiprocessing.mesh_chunk(input_list,
                                path_list,
                                core.reconstruct.create_mesh,
                                network_kwargs,
                                mesh_kwargs,
                                overwrite_meshes,
                                callback
                                )
        

    shutil.copy(os.path.join(path_deepmend,experiment_dir,'Reconstructions','ours_@inference','Meshes','0_0_.obj'),os.path.join(path_deepmend,experiment_dir,'Reconstructions_all_2','{}_0_.obj'.format(k)))
    shutil.copy(os.path.join(path_deepmend,experiment_dir,'Reconstructions','ours_@inference','Meshes','0_1_.obj'),os.path.join(path_deepmend,experiment_dir,'Reconstructions_all_2','{}_1_.obj'.format(k)))
    shutil.copy(os.path.join(path_deepmend,experiment_dir,'Reconstructions','ours_@inference','Meshes','0_2_.obj'),os.path.join(path_deepmend,experiment_dir,'Reconstructions_all_2','{}_2_.obj'.format(k)))


    #break
    #shutil.copy(os.path.join(path_deepmend,experiment_dir,'Reconstructions','ours_@inference','Meshes','0_0_.obj'),os.path.join(path_deepmend,experiment_dir,'Reconstructions_end','{}_0_.obj'.format(k)))
    #shutil.copy(os.path.join(path_deepmend,experiment_dir,'Reconstructions','ours_@inference','Meshes','0_1_.obj'),os.path.join(path_deepmend,experiment_dir,'Reconstructions_end','{}_1_.obj'.format(k)))
    #shutil.copy(os.path.join(path_deepmend,experiment_dir,'Reconstructions','ours_@inference','Meshes','0_2_.obj'),os.path.join(path_deepmend,experiment_dir,'Reconstructions_end','{}_2_.obj'.format(k)))

    
    #if k==1:
    #    break

