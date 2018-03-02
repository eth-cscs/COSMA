# - Generate a cppcheck documentation for a project.
# The function GENERATE_CPPCHECK is provided to create a "cppcheck" target that
# performs static code analysis using the cppcheck utility program.
#
# GENERATE_CPPCHECK(SOURCES <sources to check...>
#                   SOURCEDIRS <source directories>
#                   [SUPPRESSION_FILE <file>]
#                   [ENABLE_IDS <id...>]
#                   [TARGET_NAME <name>]
#                   [INCLUDES <dir...>])
#
# Generates a target "cppcheck" that executes cppcheck on the specified sources.
# Sources may either be file names or directories containing files where all
# C++ files will be parsed automatically. Use directories whenever possible
# because there is a limitation in arguments to pass to the cppcheck binary.
# SUPPRESSION_FILE may be give additionally to specify suppressions for#
# cppcheck. The sources mentioned in the suppression file must be in the same
# format like given for SOURCES. This means if you specified them relative to
# CMAKE_CURRENT_SOURCE_DIR, then the same relative paths must be used in the
# suppression file.
# When SOURCEDIRS is given, cppcheck-htmlreport is run on the produced xml output. 
# Python 2.7 with pygments is required for this feature.
# ENABLE_IDS allows to specify which additional cppcheck check ids to execute,
# e.g. all or style. They are combined with AND.
# With TARGET_NAME a different name for the generated check target can be
# specified. This is useful if several calles to this function are made in one
# CMake project, as otherwise the target names collide.
# Additional include directories for the cppcheck program can be given with
# INCLUDES.
#
# cppcheck will be executed with CMAKE_CURRENT_SOURCE_DIR as working directory.
#
# This function can always be called, even if no cppcheck was found. Then no
# target is created.
#
# Copyright (C) 2011 by Johannes Wienke <jwienke at techfak dot uni-bielefeld dot de>
#
# This program is free software; you can redistribute it
# and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation;
# either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# Modified 28-08-2014 E.J Boks (Kiwanda Embedded Systemen) to run on CMake
# 3.0.1 in conjunction with cppcheck-htmlreport
# Modified 16-02-2017 T Notargiacomo (CSCS)

get_filename_component(GENERATE_CPPCHECK_MODULE_DIR
  ${CMAKE_CURRENT_LIST_FILE} PATH)

find_package(cppcheck)

function(GENERATE_CPPCHECK)
  set(options )
  set(oneValueArgs SUPPRESSION_FILE TARGET_NAME PROJECT_NAME)
  set(multiValueArgs ENABLE_IDS SOURCES SOURCEDIRS INCLUDES)

  if (CPPCHECK_FOUND)
    CMAKE_PARSE_ARGUMENTS(ARG "${options}" "${oneValueArgs}"
      "${multiValueArgs}" ${ARGN})
    set(TARGET_NAME "cppcheck")
    set(TARGET_NAME_SUFFIX "")
    # parse target name
    list(LENGTH ARG_TARGET_NAME TARGET_NAME_LENGTH)
    if (${TARGET_NAME_LENGTH} EQUAL 1)
      set(TARGET_NAME ${ARG_TARGET_NAME})
      set(TARGET_NAME_SUFFIX "-${ARG_TARGET_NAME}")
    endif ()
        
    set(CPPCHECK_CHECKFILE
      "${CMAKE_CURRENT_BINARY_DIR}/cppcheck-files${TARGET_NAME_SUFFIX}")
    set(CPPCHECK_REPORT_FILE
      "${CMAKE_CURRENT_BINARY_DIR}/cppcheck-report${TARGET_NAME_SUFFIX}.xml")
    set(CPPCHECK_WRAPPER_SCRIPT
      "${CMAKE_CURRENT_BINARY_DIR}/cppcheck${TARGET_NAME_SUFFIX}.cmake")
     
    # write a list file containing all sources to check for the call to
    # cppcheck
    if (ARG_SOURCES)
      set(SOURCELIST ${ARG_SOURCES})
    else (ARG_SOURCES)
      set(SOURCELIST ${ARG_SOURCEDIRS})
    endif (ARG_SOURCES)
    set(SOURCE_ARGS "")
    foreach(SOURCE ${SOURCELIST})
      #Get absolute path of file, for CI tool report publication for instance
      get_filename_component(CURRENT_SOURCE_ABS_PATH ${SOURCE} ABSOLUTE)
      list(APPEND SOURCE_ARGS ${CURRENT_SOURCE_ABS_PATH})
    endforeach()
        
    # prepare a cmake wrapper to write the stderr output of cppcheck to
    # the result file
      
    # suppression argument
    list(LENGTH ARG_SUPPRESSION_FILE SUPPRESSION_FILE_LENGTH)
    if (${SUPPRESSION_FILE_LENGTH} EQUAL 1)
      get_filename_component(ABS "${ARG_SUPPRESSION_FILE}" ABSOLUTE)
      message(STATUS "Using suppression file ${ABS}")
      set(SUPPRESSION_ARGUMENT --suppressions)
      set(SUPPRESSION_FILE "\"${ABS}\"")
    endif()
        
    # includes
    set(INCLUDE_ARGUMENTS "")
    foreach(INCLUDE ${ARG_INCLUDES})
      set(INCLUDE_ARGUMENTS "${INCLUDE_ARGUMENTS} \"-I${INCLUDE}\"")
    endforeach()
        
    # enabled ids
    set(ID_LIST "")
    foreach(ID ${ARG_ENABLE_IDS})
      set(ID_LIST "${ID_LIST},${ID}")
    endforeach()
    if (ID_LIST)
      string(LENGTH ${ID_LIST} LIST_LENGTH)
      math(EXPR FINAL_LIST_LENGTH "${LIST_LENGTH} - 1")
      string(SUBSTRING ${ID_LIST} 1 ${FINAL_LIST_LENGTH} FINAL_ID_LIST)
      set(IDS_ARGUMENT "\"--enable=${FINAL_ID_LIST}\"")
    else()
      set(IDS_ARGUMENT "")
    endif()
     
    if (ARG_SOURCEDIRS)
      set(CPPCHECK_REPORT_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/cppcheckdir-report${TARGET_NAME_SUFFIX}")   
      file(WRITE ${CPPCHECK_WRAPPER_SCRIPT}
        "EXECUTE_PROCESS(COMMAND \"${CPPCHECK_EXECUTABLE}\"
          ${INCLUDE_ARGUMENTS} ${SUPPRESSION_ARGUMENT} ${SUPPRESSION_FILE}
          ${IDS_ARGUMENT} --inline-suppr --xml ${SOURCE_ARGS}
          RESULT_VARIABLE CPPCHECK_EXIT_CODE
          ERROR_VARIABLE ERROR_OUT
          WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}\")
          if (NOT CPPCHECK_EXIT_CODE EQUAL 0)
            message(FATAL_ERROR \"Error executing cppcheck for target
              ${TARGET}, return code: \${CPPCHECK_EXIT_CODE}\")
          else (NOT CPPCHECK_EXIT_CODE EQUAL 0)
            file(WRITE \"${CPPCHECK_REPORT_FILE}\" \"\${ERROR_OUT}\")
            EXECUTE_PROCESS(COMMAND \"${CPPCHECK_EXECUTABLE}-htmlreport\"
            \"--file=${CPPCHECK_REPORT_FILE}\" 
            \"--report-dir=${CPPCHECK_REPORT_DIR}\"
            \"--source-dir=${SOURCE_ARGS}\"
            \"--title=${ARG_PROJECT_NAME}\")
          endif ()
          if (ERROR_OUT)
            message(\"Detected errors:\\n\${ERROR_OUT}\")
          endif ()
        ")
    else (ARG_SOURCEDIRS)
      file(WRITE ${CPPCHECK_WRAPPER_SCRIPT}
        "EXECUTE_PROCESS(COMMAND \"${CPPCHECK_EXECUTABLE}\"
           ${INCLUDE_ARGUMENTS} ${SUPPRESSION_ARGUMENT} ${SUPPRESSION_FILE}
           ${IDS_ARGUMENT} --inline-suppr --xml-version=2 --xml ${SOURCE_ARGS}
           RESULT_VARIABLE CPPCHECK_EXIT_CODE
           ERROR_VARIABLE ERROR_OUT
           WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}\")
           if (NOT CPPCHECK_EXIT_CODE EQUAL 0)
             message(FATAL_ERROR \"Error executing cppcheck for target
               ${TARGET}, return code: \${CPPCHECK_EXIT_CODE}\")
           endif ()
           if (ERROR_OUT)
             message(\"Detected errors:\\n\${ERROR_OUT}\")
           endif ()
           file(WRITE \"${CPPCHECK_REPORT_FILE}\" \"\${ERROR_OUT}\")
             ")
    endif (ARG_SOURCEDIRS)
    
    ADD_CUSTOM_TARGET(${TARGET_NAME} ${CMAKE_COMMAND} -P
      "${CPPCHECK_WRAPPER_SCRIPT}"
      COMMENT "Generating cppcheck result ${TARGET_NAME}")      
    message(STATUS "Generating cppcheck target with name ${TARGET_NAME}")
  endif ()
endfunction()

