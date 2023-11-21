import sys
sys.path.append('/mnt/home/jsaragih/code/codec_body/train/')
#from body_vae_new import linear_blend_skinning as LBSfunc
#from body_vae_new.linear_blend_skinning import LinearBlendSkinning
#from linear_blend_skinning_cuda import LinearBlendSkinningCuda 
#import plyutils
#import objutils
#import krtutils
#from plyutils import LoadPly, SavePly



# source /mnt/home/gbschwartz/anaconda/bin/activate py3_pytorch_1.0
import sys
import numpy as np
import os

#from plyutils import LoadPly, SavePly
from objutils import load_obj, write_obj

#    return v, vt, vindices, vtindices

#def write_obj(path, v, vt, vi, vti, vn=None, vc=None):

################################################################################
def interpolate_bag_vertices(interior_vi, source_vi, source_bary, pV):
    # pV -> Nx3 mesh vertices on which to interpolate the bag
    # interior_vi - [vertex_id] len()==Ni list of vertex indices to replace
    # source_vi - Nix3 vertex indices of vertices used during barycentric interpolation
    # source_bary - Nx3 barycentric weights for each source_vi vertex
    pVi = pV.copy()
    pVi[interior_vi,:] = np.sum(pV[source_vi,:]*source_bary[:,:,None],axis=1)
    return pVi
################################################################################
if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Interpolate bag vertices on mesh')
    parser.add_argument('files', type=str, nargs='+', help='input mesh files')
    parser.add_argument(
        '--out_path',
        default='/mnt/captures/tsimon/devel/riglord/temp/',
        type=str,
        help='output base path')
    parser.add_argument(
        '--interp_data_path',
        default='/mnt/captures/tsimon/devel/riglord/v1/mesh-closer.npz',
        type=str,
        help='interpolation file')

    args = parser.parse_args()
    out_path = args.out_path
    interp_data_path = args.interp_data_path

    files = args.files

    # Load interpolation data
    data = np.load(interp_data_path)
    interior_vi = data['interior_vi']
    source_vi = data['source_vi']
    source_bary = data['source_bary']

    for fpath in files:
        #pV, vc, vt, vi, vti = LoadPly(fpath)
        pV, vt, vi, vti = load_obj(fpath)
        pV = np.asarray(pV)
        vt = np.asarray(vt)
        vi = np.asarray(vi)
        vti = np.asarray(vti)
        pVi = interpolate_bag_vertices(interior_vi, source_vi, source_bary, pV)

        ofpath = os.path.join(out_path,os.path.basename(fpath))
        print(ofpath)
        #SavePly(ofpath, pVi, vc,vt,vi,vti)              
        write_obj(ofpath, pVi, vt, vi, vti)
