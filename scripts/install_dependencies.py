#!/usr/bin/env python3 
import argparse
import os
import sys
import tempfile
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument(
        'prefix', 
        type=str,
        help='Installation prefix for dependencies'
        )
args = parser.parse_args()
if not os.path.isdir(args.prefix):
    print("The argument is not a directory.")
    sys.exit()


def install_lib(tmppath, prefix, libname):
    url = 'https://github.com/kabicm/{libname}.git'.format(**locals())
    clone_dir = os.path.join(tmppath, libname)
    build_dir = os.path.join(tmppath, 'build_{libname}'.format(**locals()))
    install_dir ='{prefix}/{libname}-master'.format(**locals())

    config_cmd = ('cmake ../{libname} '
                    '-DCMAKE_BUILD_TYPE=Release '
                    '-DCMAKE_INSTALL_PREFIX={install_dir}'.format(**locals())
                 )
    build_and_install_cmd = 'cmake --build . --target install'
    os.system('git clone --recursive {url} {clone_dir}'.format(**locals()))
    os.makedirs(build_dir, exist_ok=True)
    subprocess.call(config_cmd, cwd=build_dir, shell=True)
    subprocess.call(build_and_install_cmd, cwd=build_dir, shell=True)

    return install_dir

with tempfile.TemporaryDirectory() as tmppath:
    install_dirs = ''
    for libname in ['options', 'semiprof', 'grid2grid']:
        install_dirs += '{};'.format(install_lib(tmppath, args.prefix, libname)) 

    print('\nUse the following CMake parameter: -DCMAKE_PREFIX_PATH="{}"'.format(install_dirs))
