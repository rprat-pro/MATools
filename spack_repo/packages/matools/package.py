from spack import *

class Matools(AutotoolsPackage):
    """MATools is a library that offers various tools, including MATimers (timers in hierarchical form), MATrace (Trace generation for VITE), and MAMemory (memory footprint printing).
		"""

    homepage = "https://github.com/rprat-pro/MATools/"
    git = "https://github.com/rprat-pro/MATools.git"


    version("1.0", mat100="ae480a40004560fe26e7c7be05337d1290a1ad60")

    depends_on("cmake")
    depends_on("mpi", when="+mpi")
    depends_on("vite", when="+trace")

    build_system("cmake", "autotools", default="cmake")

    variant("mpi", default=False, description="Support for MPI")
    variant("trace", default=False, description="add install for VITE trace")


    def cmake_args(self):
        args = [
            self.define_from_variant("MATOOLS_MPI", "mpi"),
        ]
        return args
