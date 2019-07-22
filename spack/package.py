from spack import *

class Cosma(CMakePackage, CudaPackage):
    """Distributed Communication-Optimal Matrix-Matrix Multiplication Library"""

    homepage = "https://github.com/eth-cscs/COSMA"
    version('develop', 
            git='https://github.com/eth-cscs/COSMA.git', 
            branch='master',
            submodules=True)

    variant('build_type', default='Release',
        description='CMake build type',
        values=('Debug', 'Release', 'RelWithDebInfo', 'MinSizeRel'))

    variant('scalapack', default=False,
            description='Support for conversion from/to ScaLAPACK P?GEMM.')

    depends_on('cmake@3.12:', type='build')
    depends_on('mpi@3:')
    depends_on('intel-mkl threads=openmp')
    depends_on('scalapack', when='scalapack=True')

    def cmake_args(self): 
        spec = self.spec
        args = []

        if spec.satisfies('%gcc'):
            args.append('-DMKL_THREADING=GOMP')
        else:
            args.append('-DMKL_THREADING=IOMP')

        if spec['mpi'].name == 'openmpi':
            args.append('-DMKL_MPI_TYPE=OMPI')
        else:
            args.append('-DMKL_MPI_TYPE=MPICH')

        return args
