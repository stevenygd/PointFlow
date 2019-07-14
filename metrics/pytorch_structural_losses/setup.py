from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Python interface
setup(
    name='PyTorchStructuralLosses',
    version='0.1.0',
    install_requires=['torch'],
    packages=['StructuralLosses'],
    package_dir={'StructuralLosses': './'},
    ext_modules=[
        CUDAExtension(
            name='StructuralLossesBackend',
            include_dirs=['./'],
            sources=[
                'pybind/bind.cpp',
            ],
            libraries=['make_pytorch'],
            library_dirs=['objs'],
            # extra_compile_args=['-g']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Christopher B. Choy',
    author_email='chrischoy@ai.stanford.edu',
    description='Tutorial for Pytorch C++ Extension with a Makefile',
    keywords='Pytorch C++ Extension',
    url='https://github.com/chrischoy/MakePytorchPlusPlus',
    zip_safe=False,
)
