macro(add_cuda_lib_executable exename)
  cuda_add_executable(${exename} "${exename}.cu")
  set(dependencyName ${ARGN})
  target_link_libraries(${exename} ${dependencyName} )
endmacro()

macro(add_lib_executable exename)
  add_executable(${exename} "${exename}.cpp")
  set(dependencyName ${ARGN})
  target_link_libraries(${exename} ${dependencyName} )
endmacro()

macro(cuda_add_lib_executable exename)
  cuda_add_executable(${exename} "${exename}.cpp")
  set(dependencyName ${ARGN})
  target_link_libraries(${exename} ${dependencyName} )
endmacro()

# Sets a variable and adds it in the CMake cache.
# and initializes to default if variable is not defined.
function(setoption variable type default description)
  if (DEFINED ${variable})
    set_property(CACHE "${variable}" PROPERTY TYPE "${type}")
    set_property(CACHE "${variable}" PROPERTY HELPSTRING "${description}")
  else ()
    set(${variable} ${default} CACHE "${type}" "${description}")
  endif()
endfunction()
