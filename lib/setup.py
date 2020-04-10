import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "csrc")
    
    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    
    sources = main_file + source_cuda
    
    extension = CUDAExtension
    
    sources = [os.path.join(extensions_dir, s) for s in sources]

    ext_modules = [
        extension("_C", sources)
    ]

    return ext_modules

setup(
    name='faster_rcnn_utils',
    version='0.1',
    description='Faster RCNN CUDA utilities',
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension
    }
)