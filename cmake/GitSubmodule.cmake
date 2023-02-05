# Call to ensure that the git submodule in location `path` is loaded.
# If the submodule is not loaded, an error message that describes
# how to update the submodules is printed.
# Sets the variable name_avail to `ON` if the submodule is available,
# or `OFF` otherwise.
# copyright github.com/arbor-sim

function(check_git_submodule name path)
    set(success_var "${name}_avail")
    set(${success_var} ON PARENT_SCOPE)

    get_filename_component(dotgit "${path}/.git" ABSOLUTE)
    if(NOT EXISTS ${dotgit})
        message(
            "\nThe git submodule for ${name} is not available.\n"
            "To check out all submodules use the following commands:\n"
            "    git submodule init\n"
            "    git submodule update\n"
            "Or download submodules recursively when checking out:\n"
            "    git clone --recursive https://github.com/eth-cscs/COSMA.git\n"
        )

        # if the repository was not available, and git failed, set AVAIL to false
        set(${success_var} OFF PARENT_SCOPE)
    endif()
endfunction()

function(add_git_submodule_or_find_external name path)
  check_git_submodule(${name} "libs/cxxopts")
  if(NOT ${name}_avail)
    # attempt to find system installation of pybind11
    find_package(${name} REQUIRED)
  else()
    message(VERBOSE "Using ${name} as git submodule from ${path}")
    add_subdirectory("${path}")
  endif()
endfunction()
