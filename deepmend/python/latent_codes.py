import json
import os
import open3d as o3d
import igl
import fnmatch
import argparse

import torch

import core

import core.workspace as ws





def find_objects(pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result



def save_latent_codes(dir_old,dir_new,dir_dict,dir_specs):

    ''' in old_dir: directory, where th latent code is saved; latent+late_tool is saved in one tensor
        dir_new save latent codes in new directory, seperately in latent code and tool latent code
        dir_dict: directory, where the the trai/test split dictionary can be found
        dir_specs: directory, where the specification of the network can be found 
    
    '''

    f=open(os.path.join(dir_dict,'mugs_split.json'))
    dict=json.load(f)['id_test_list']

    latent_code=torch.load('/home/michael/Projects/DeepMend_changes3/deepmend/experiments/mugs/Reconstructions/deepmend@latest/Codes/0_1000_3000_0.0005_0.0001_0.001_0.005_.pth')
    
    specs=ws.load_experiment_specifications(dir_specs)
    
    if latent_code[0].size()[0]!=specs['CodeLength']+specs['BreakCodeLength']:
        return 'Error: Size of Latent Code does not not fit with Network Architecture'
    
        
    all_obj=find_objects('*.pth',dir_old)
    len_path=len(dir_old)

    for k,id in enumerate(dict):

        if k<10:
            path_old=[i for i in all_obj if '{}_'.format(k) in i[len_path_recon:len_path_recon+2]][0]
               
        
        if k>=10:
            path_old=[i for i in all_obj if '{}_'.format(k) in i[len_path_recon:len_path_recon+3]][0]

        
        weight=torch.load(path_old)


        dict={'epoch':0,'latent_codes':{'weight':None }}
        dict['latent_codes']['weight']=weight[:,:1000]
        torch.save(dict,os.path.join(dir_new,'{}.pth'.format(k)))

        dict={'epoch':0,'latent_codes':{'weight':None }}
        dict['latent_codes']['weight']=weight[:,1000:]
        torch.save(dict,os.path.join(dir_new,'{}_tool.pth'.format(k)))

        



                
        
        


if __name__ == "__main__":





    arg_parser_latent = argparse.ArgumentParser(description="Save Latent Codes")
    arg_parser_latent.add_argument(
        "--dir_old",
         "-old",
         default="/home/michael/Projects/Reconstructions/deepmend@2000/Codes",
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
         default="/home/michael/Projects/test_shapeNet/ShapeNet",
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