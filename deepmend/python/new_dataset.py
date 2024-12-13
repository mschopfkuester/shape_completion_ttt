import json
import os
import open3d as o3d
import igl
import fnmatch
import argparse

import numpy as np

import torch

import core

import core.workspace as ws

import shutil




def find_objects(pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result



def create_new_dataset(datadir_old,datadir_new,path_recon_obj):


    samp='model_0_sampled.npz'

    b_obj='model_b_0.obj'
    b_sdf='model_b_0_partial_sdf.npz'
    b_sdf_only='model_b_0_sdf.npz'
    b_occ='model_b_0_uniform_occ.npz'
    b_occ_uni='model_b_0_uniform_padded_occ_32.npz'


    r_obj='model_r_0.obj'
    r_sdf='model_r_0_sdf.npz'
    r_occ='model_r_0_uniform_occ.npz'
    r_occ_uni='model_r_0_uniform_padded_occ_32.npz'

    

    c_obj='model_c.obj'
    

    spline='model_spline_0_sdf.npz'
    plane='model_spline_0_plane.ply'



    
    f=open(os.path.join(datadir_old,'split_original','mugs_split.json'))
    dict=json.load(f)['id_test_list']

    

    for k,id in enumerate(dict):
        print(len(dict))
        print(id)
        ####load original fractured shape in new dataset
        shutil.copy(os.path.join(datadir_old,id[0],id[1],'models',r_obj),os.path.join(datadir_new,id[0],id[1],'models',r_obj))
        continue
        
        try:
            os.mkdir(os.path.join(datadir_new,id[0],id[1]))
            os.mkdir(os.path.join(datadir_new,id[0],id[1],'models'))
        except OSError:
            pass
        shutil.copy(os.path.join(datadir_old,id[0],id[1],'models',samp),os.path.join(datadir_new,id[0],id[1],'models',samp))

        shutil.copy(os.path.join(datadir_old,id[0],id[1],'models',spline),os.path.join(datadir_new,id[0],id[1],'models',spline))
        shutil.copy(os.path.join(datadir_old,id[0],id[1],'models',plane),os.path.join(datadir_new,id[0],id[1],'models',plane))
        shutil.copy(os.path.join(datadir_old,id[0],id[1],'models',b_obj),os.path.join(datadir_new,id[0],id[1],'models',b_obj))


        #copy all r files (we not need them, but the dataloader would not work)
        shutil.copy(os.path.join(datadir_old,id[0],id[1],'models',r_sdf),os.path.join(datadir_new,id[0],id[1],'models',r_sdf))
        shutil.copy(os.path.join(datadir_old,id[0],id[1],'models',r_occ),os.path.join(datadir_new,id[0],id[1],'models',r_occ))
        shutil.copy(os.path.join(datadir_old,id[0],id[1],'models',r_occ_uni),os.path.join(datadir_new,id[0],id[1],'models',r_occ_uni))

        
        #shutil.copy(os.path.join(datadir_old,id[0],id[1],'models',b_sdf),os.path.join(datadir_new,id[0],id[1],'models',b_sdf))
        #shutil.copy(os.path.join(datadir_old,id[0],id[1],'models',b_sdf_only),os.path.join(datadir_new,id[0],id[1],'models',b_sdf_only))
        #shutil.copy(os.path.join(datadir_old,id[0],id[1],'models',b_occ),os.path.join(datadir_new,id[0],id[1],'models',b_occ))
        #shutil.copy(os.path.join(datadir_old,id[0],id[1],'models',b_occ_uni),os.path.join(datadir_new,id[0],id[1],'models',b_occ_uni))


        #shutil.copy(os.path.join(datadir_old,id[0],id[1],'models/model_r_0.obj'),os.path.join(datadir_new,id[0],id[1],'models/model_r_0.obj'))
        
        ####load reconstructed complete+reconstruction shape

        #find path for recon + complete object
        all_obj=find_objects('*.obj',path_recon_obj)
        len_path_recon=len(path_recon_obj)
        
        if k<10:
            #print([ i[len_path_recon+1:len_path_recon+4] for i in all_obj ])
            complete_path=[i for i in all_obj if '{}_0'.format(k) in i[len_path_recon+1:len_path_recon+4]][0]
            rest_path=[i for i in all_obj if '{}_2'.format(k) in i[len_path_recon+1:len_path_recon+4]][0]    
        
        if k>=10:
            complete_path=[i for i in all_obj if '{}_0'.format(k) == i[len_path_recon+1:len_path_recon+5]][0]
            rest_path=[i for i in all_obj if '{}_2'.format(k) == i[len_path_recon+1:len_path_recon+5]][0]

        shutil.copy(complete_path,os.path.join(datadir_new,id[0],id[1],'models',c_obj))
        shutil.copy(rest_path,os.path.join(datadir_new,id[0],id[1],'models',r_obj))

        
        


def save_latent_codes(dir_old,dir_new,dir_dict,dir_specs):

    ''' in old_dir: directory, where th latent code is saved; latent+late_tool is saved in one tensor
        dir_new save latent codes in new directory, seperately in latent code and tool latent code
        dir_dict: directory, where the the trai/test split dictionary can be found
        dir_specs: directory, where the specification of the network can be found 
    
    '''

    f=open(os.path.join(dir_dict,'mugs_split.json'))
    dict=json.load(f)['id_test_list']
    test_id=[]

    for id in dict:
        try:
            np.load('/home/michael/Projects/test_shapeNet/ShapeNet/ShapeNetCore.v2/03797390/{}/models/model_b_0_uniform_occ.npz'.format(id[1]))
            
        except FileNotFoundError:
            c=0
        else:
            test_id.append(id)


    
    specs=ws.load_experiment_specifications(dir_specs)
 
        
    all_obj=find_objects('*.pth',dir_old)
    len_path_recon=len(dir_old)

    for k,id in enumerate(test_id):
        print(test_id)

        if k<10:
            print([i[len_path_recon+1:len_path_recon+3] for i in all_obj ])
            path_old=[i for i in all_obj if '{}_'.format(k) in i[len_path_recon+1:len_path_recon+3]][0]
               
        
        if k>=10:
            path_old=[i for i in all_obj if '{}_'.format(k) in i[len_path_recon+1:len_path_recon+4]][0]

        
        weight=torch.load(path_old)


        dict={'epoch':0,'latent_codes':{'weight':None }}
        dict['latent_codes']['weight']=weight[:,:1000]
        torch.save(dict,os.path.join(dir_new,'{}.pth'.format(k)))

        dict={'epoch':0,'latent_codes':{'weight':None }}
        dict['latent_codes']['weight']=weight[:,1000:]
        torch.save(dict,os.path.join(dir_new,'{}_tool.pth'.format(k)))

        



                
        
        


if __name__ == "__main__":


    

    arg_parser_dataset = argparse.ArgumentParser(description="Create New Dataset")
    arg_parser_dataset.add_argument(
        "--datadir_old",
        "-old",
        default="/home/michael/Projects/test_shapeNet/ShapeNet/ShapeNetCore.v2",
        help= "Old Datadir, eventually saved as export DATADIR",
    )
    arg_parser_dataset.add_argument(
        "--datadir_new",
        "-new",
        default="/home/michael/Projects/test_shapeNet_inference2/ShapeNet/ShapeNetCore.v2",
        help="New Datadir, eventually saved as export DATADIR2",
    )    
    arg_parser_dataset.add_argument(
        "--path_recon_obj",
        "-obj",
        default="/home/michael/Projects/DeepMend_changes3_new2/deepmend/experiments/mugs/Reconstructions/deepmend@b/Meshes",
        help="Path to reconstructed objects",
    )        
    core.add_common_args(arg_parser_dataset)
    args = arg_parser_dataset.parse_args()
    core.configure_logging(args)

    create_new_dataset(args.datadir_old,args.datadir_new,args.path_recon_obj)



    arg_parser_latent = argparse.ArgumentParser(description="Save Latent Codes")
    arg_parser_latent.add_argument(
         "--dir_old",
         "-old",
         default="/home/michael/Projects/DeepMend_changes3_new2/deepmend/experiments/mugs/Reconstructions/deepmend@b/Codes",
         help= "Old Datadir, eventually saved as export DATADIR",
     )

    arg_parser_latent.add_argument(
         "--dir_new",
         "-new",
         default="/home/michael/Projects/DeepMend_changes3_new2/deepmend/experiments/mugs/LatentCodesInference1",
         help= "Old Datadir, eventually saved as export DATADIR",
     )

    arg_parser_latent.add_argument(
         "--dir_dict",
         "-dict",
         default="/home/michael/Projects/test_shapeNet/ShapeNet/ShapeNetCore.v2",
         help= "Old Datadir, eventually saved as export DATADIR",
     )

    arg_parser_latent.add_argument(
         "--dir_specs",
         "-specs",
         default="/home/michael/Projects/DeepMend_changes3_new2/deepmend/experiments/mugs",
         help= "Old Datadir, eventually saved as export DATADIR",
     )



    core.add_common_args(arg_parser_latent)
    args = arg_parser_latent.parse_args()
    core.configure_logging(args)

    #save_latent_codes(args.dir_old,args.dir_new,args.dir_dict,args.dir_specs)