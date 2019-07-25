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

    variant('tiledmm', default=False,
            description='Use Tiled-MM GPU back end.')

    depends_on('cmake@3.12:', type='build')
    depends_on('mpi@3:')
    depends_on('intel-mkl threads=openmp', when='tiledmm=False')
    depends_on('scalapack', when='tiledmm=False')
    depends_on('cuda', when='tiledmm=True')

    def cmake_args(self):
        spec = self.spec
        args = ['-DCOSMA_WITH_TESTS=OFF',
                '-DCOSMA_WITH_APPS=OFF',
                '-DCOSMA_WITH_BENCHMARKS=OFF']

        if '+tiledmm' in spec:
            args.append('-DCOSMA_WITH_TILEDMM=ON')

        if spec['mpi'].name == 'openmpi':
            args.append('-DMKL_MPI_TYPE=OMPI')
        
        return args
