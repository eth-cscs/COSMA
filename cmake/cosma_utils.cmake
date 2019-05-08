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
