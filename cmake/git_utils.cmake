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
            "    git clone --recursive https://github.com/electronic-structure/SIRIUS.git\n"
            )

        # if the repository was not available, and git failed, set AVAIL to false
        set(${success_var} OFF PARENT_SCOPE)
    endif()
endfunction()
