from spack import *

class Matools(CMakePackage):
    """MATools is a library that offers various tools, including MATimers (timers in hierarchical form), MATrace (Trace generation for VITE), and MAMemory (memory footprint printing).
		"""

    homepage = "https://github.com/rprat-pro/MATools/"
    git = "https://github.com/rprat-pro/MATools.git"

    version("1.2", commit="1d071628dd7b13971c45ed00e8a00c1e553003af")
    version("1.1", commit="ea62ca75455d0741a6a179114734e9b12fa9f412")
    version("1.0", commit="439f19525e10bae163da68abb00eed4203951af4")

    depends_on("cmake")
    depends_on("openmpi", when="+mpi")
    depends_on("vite", when="+trace")

    build_system("cmake", default="cmake")

    variant("mpi", default=False, description="Support for MPI")
    variant("static", default=False, description="Using static library")
    variant("trace", default=False, description="add install for VITE trace")


    def cmake_args(self):
        args = [
            self.define_from_variant("MATOOLS_MPI", "mpi"),
            self.define_from_variant("MATOOLS_STATIC_LIB", "static"),
        ]
        return args
