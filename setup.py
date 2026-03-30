import os
import re
import subprocess

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

SETUP_DIR = os.path.dirname(os.path.abspath(__file__))
NATURALNEIGHBOR_PKG = os.path.join(SETUP_DIR, "naturalneighbor")

module = Extension(
    "cnaturalneighbor",
    include_dirs=[numpy.get_include()],
    library_dirs=["/usr/local/lib"],
    extra_compile_args=["--std=c++11", "-O3"],
    sources=[
        "naturalneighbor/cnaturalneighbor.cpp",
    ],
)


def find_cuda_home():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.isdir(cuda_home):
        return cuda_home
    if os.path.isdir("/usr/local/cuda"):
        return "/usr/local/cuda"
    return None


def nvcc_arch_flags(cuda_home):
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc):
        return None
    try:
        out = subprocess.check_output([nvcc, "--version"], stderr=subprocess.STDOUT, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    m = re.search(r"release\s+(\d+)\.(\d+)", out, re.IGNORECASE)
    if not m:
        return None
    major = int(m.group(1))
    cuda_12_archs = [
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_89,code=compute_89",
    ]
    cuda_13_archs = [
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_89,code=compute_89",
    ]
    if major >= 13:
        return cuda_13_archs
    return cuda_12_archs


class BuildExtCUDA(build_ext):
    def compile_cuda_object(self, cu_source, cuda_home, arch_flags):
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        self.mkpath(self.build_temp)
        cu_abs = os.path.abspath(cu_source)
        base = os.path.splitext(os.path.basename(cu_abs))[0]
        obj = os.path.join(self.build_temp, base + ".cu.o")
        cmd = [
            nvcc,
            "-c",
            cu_abs,
            "-o",
            obj,
            "-I",
            numpy.get_include(),
            "-I",
            NATURALNEIGHBOR_PKG,
            "-I",
            os.path.join(cuda_home, "include"),
            "-O3",
            "--std=c++14",
            "--compiler-options",
            "-fPIC",
        ] + arch_flags
        self.spawn(cmd)
        return obj

    def build_extensions(self):
        cuda_home = find_cuda_home()
        arch_flags = nvcc_arch_flags(cuda_home) if cuda_home else None

        for ext in self.extensions:
            if ext.name != "ccudanaturalneighbor":
                continue
            if not cuda_home or not arch_flags:
                raise RuntimeError(
                    "ccudanaturalneighbor requires CUDA toolkit 12.4+ (set CUDA_HOME or install "
                    "under /usr/local/cuda)."
                )
            cu_sources = [s for s in ext.sources if s.endswith(".cu")]
            cpp_sources = [s for s in ext.sources if not s.endswith(".cu")]
            extra_objects = list(getattr(ext, "extra_objects", []) or [])
            for cu in cu_sources:
                extra_objects.append(self.compile_cuda_object(cu, cuda_home, arch_flags))
            ext.sources = cpp_sources
            ext.extra_objects = extra_objects
            lib_dirs = list(ext.library_dirs or [])
            for sub in ("lib64", "lib"):
                p = os.path.join(cuda_home, sub)
                if os.path.isdir(p) and p not in lib_dirs:
                    lib_dirs.append(p)
            ext.library_dirs = lib_dirs
            libs = list(ext.libraries or [])
            if "cudart" not in libs:
                libs.append("cudart")
            ext.libraries = libs

        super().build_extensions()


def make_extensions():
    exts = [module]
    cuda_home = find_cuda_home()
    arch_flags = nvcc_arch_flags(cuda_home) if cuda_home else None
    if cuda_home and arch_flags:
        cuda_module = Extension(
            "ccudanaturalneighbor",
            include_dirs=[
                numpy.get_include(),
                NATURALNEIGHBOR_PKG,
                os.path.join(cuda_home, "include"),
            ],
            library_dirs=[
                os.path.join(cuda_home, "lib64"),
                os.path.join(cuda_home, "lib"),
            ],
            libraries=["cudart"],
            extra_compile_args=["--std=c++11", "-O3"],
            sources=[
                "naturalneighbor/cnaturalneighbor_cuda.cpp",
                "naturalneighbor/cuda_kernels.cu",
            ],
        )
        exts.append(cuda_module)
    return exts


extensions = make_extensions()
cmdclass = {"build_ext": BuildExtCUDA} if any(e.name == "ccudanaturalneighbor" for e in extensions) else {}

setup(
    name="naturalneighbor",
    version="0.2.2",
    description="Fast, discrete natural neighbor interpolation in 3D on a CPU.",
    long_description=open("README.rst", "r").read(),
    author="Reece Stevens",
    author_email="rstevens@innolitics.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    keywords="interpolation scipy griddata numpy sibson",
    install_requires=[
        "numpy>=1.13",
    ],
    packages=["naturalneighbor"],
    ext_modules=extensions,
    cmdclass=cmdclass,
)
