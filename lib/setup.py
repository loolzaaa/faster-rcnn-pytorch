import os
import glob
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "csrc")
    
    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    
    sources = main_file + source_cpu
    extension = CppExtension

    define_macros = []

    if torch.cuda.is_available():
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
    
    sources = [os.path.join(extensions_dir, s) for s in sources]

    ext_modules = [
        extension("_C", sources, define_macros=define_macros)
    ]

    return ext_modules

setup(
    name='faster_rcnn_utils',
    version='0.3',
    description='Faster RCNN CPU/CUDA utilities',
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension
    }
)