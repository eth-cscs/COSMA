from spack import *

class Cosma(CMakePackage, CudaPackage):
    """Distributed Communication-Optimal Matrix-Matrix Multiplication Library"""

    homepage = "https://github.com/eth-cscs/COSMA"
    version('develop', 
            git='https://github.com/eth-cscs/COSMA.git', 
            branch='master',
            submodules=True)

    variant('blas', default='mkl',
        values=('mkl', 'openblas', 'cray-libsci', 'custom', 'cuda', 'rocm'),
        description='BLAS backend')

    variant('scalapack', default='none',
            values=('mkl', 'cray-libsci', 'custom'),
            description='Optional ScaLAPACK support.')

    depends_on('cmake@3.12:', type='build')
    depends_on('mpi@3:')

    depends_on('intel-mkl', when='blas=mkl')
    depends_on('openblas', when='blas=openblas')
    depends_on('netlib-lapack', when='blas=custom')
    depends_on('netlib-scalapack', when='scalapack=custom')
    depends_on('cray-libsci', when='scalapack=cray-libsci')
    depends_on('cuda', when='blas=cuda')
    #TODO: rocm


    def setup_environment(self, spack_env, run_env):
        if 'blas=cuda' in self.spec:
            spack_env.set('CUDA_PATH', self.spec['cuda'].prefix)

    def cmake_args(self):
        spec = self.spec
        args = ['-DCOSMA_WITH_TESTS=OFF',
                '-DCOSMA_WITH_APPS=OFF',
                '-DCOSMA_WITH_PROFILING=OFF',
                '-DCOSMA_WITH_BENCHMARKS=OFF']

        if 'blas=mkl' in spec:
            args += ['-DCOSMA_BLAS=MKL']
        elif 'blas=cray-libsci' in spec:
            args += ['-DCOSMA_BLAS=CRAY_LIBSCI']
        elif 'blas=openblas' in spec:
            args += ['-DCOSMA_BLAS=OPENBLAS']
        elif 'blas=custom' in spec:
            args += ['-DCOSMA_BLAS=CUSTOM']
        elif 'blas=cuda' in spec:
            args += ['-DCOSMA_BLAS=CUDA']
        else: # 'blas=rocm' in spec:
            args += ['-DCOSMA_BLAS=ROCM']

        if 'scalapack=mkl' in spec:
            args += ['-DCOSMA_SCALAPACK=MKL']
        elif 'scalapack=cray-libsci' in spec:
            args += ['-DCOSMA_SCALAPACK=CRAY_LIBSCI']
        elif 'scalapack=custom' in spec:
            args += ['-DCOSMA_SCALAPACK=CUSTOM']

        return args
