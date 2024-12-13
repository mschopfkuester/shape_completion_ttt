import numpy as np
import torch
import igl
import os

import processor.process_sdf as sdf_compute
import processor.process_occupancies_uniform as compute_occ_uni
import processor.process_sdf_uniform as compute_sdf_uni

import json





#f,v=igl.read_triangle_mesh('/home/michael/Projects/test_shapeNet_inference2/ShapeNet/ShapeNetCore.v2/03797390/8012f52dd0a4d2f718a93a45bf780820/models/model_r_0.obj')

#print(f.min(),f.max())



def preprocessing(datadir,class_no,id):




        path=os.path.join(datadir,class_no,id,'models')
        print(path)

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
        c_sdf='model_c_0_sdf.npz'
        c_occ='model_c_0_uniform_occ.npz'
        c_occ_uni='model_c_0_uniform_padded_occ_32.npz'


        spline='model_spline_0_sdf.npz'
        plane='model_spline_0_plane.ply'






        #compute sdf values

        
        sdf_compute.process(f_in=os.path.join(path,b_obj),
                        f_out=os.path.join(path,b_sdf),
                        f_samp=os.path.join(path,samp))




        #sdf_compute.process(f_in=os.path.join(path,r_obj),
        #                f_out=os.path.join(path,r_sdf),
        #                f_samp=os.path.join(path,samp),
        #                overwrite=False)


        sdf_compute.process(f_in=os.path.join(path,c_obj),
                        f_out=os.path.join(path,c_sdf),
                        f_samp=os.path.join(path,samp))


        #for fractured: create model_b_0 -> only sdf file


        frac=np.load(os.path.join(path,b_sdf))['sdf']
        np.savez(os.path.join(path,b_sdf_only),sdf=frac)

  
        #compute occupancies unif


        compute_occ_uni.process(os.path.join(path,b_obj),
                                os.path.join(path,b_occ)
                        )

        compute_occ_uni.process(os.path.join(path,c_obj),
                                os.path.join(path,c_occ)
                        )

        #compute_occ_uni.process(os.path.join(path,r_obj),
        #                        os.path.join(path,r_occ))


        #compute occupancies unif

        compute_sdf_uni.process(os.path.join(path,b_obj),
                                os.path.join(path,b_occ_uni)
                        )

        compute_sdf_uni.process(os.path.join(path,c_obj),
                                os.path.join(path,c_occ_uni)
                        )

        #compute_sdf_uni.process(os.path.join(path,r_obj),
        #                        os.path.join(path,r_occ_uni)
        #                )
        


if __name__ == '__main__':

        train_test_file='/home/michael/Projects/test_shapeNet/ShapeNet/ShapeNetCore.v2/split_original/mugs_split_firststep_false.json'
        f=open(train_test_file)
        id_list=json.load(f)['id_test_list']

        for class_no,id in id_list:
        
        
                datadir='/home/michael/Projects/test_shapeNet_inference2/ShapeNet/ShapeNetCore.v2'
                #class_no='03797390'
                #id='24651c3767aa5089e19f4cee87249aca'

                preprocessing(datadir,class_no,id)