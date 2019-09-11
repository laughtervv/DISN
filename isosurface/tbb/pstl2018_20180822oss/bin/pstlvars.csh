#!/bin/csh
#
# Copyright (c) 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
#

# Parsing script arguments
# Arg1 represents target architecture. Its possible values are 'ia32' or 'intel64',
# default value equals to the value of $COMPILERVARS_ARCHITECTURE environment variable.

set PSTL_TARGET_ARCH=""

if ($?COMPILERVARS_ARCHITECTURE) then
    set PSTL_TARGET_ARCH="$COMPILERVARS_ARCHITECTURE"
endif

if ("$1" != "") then
    set PSTL_TARGET_ARCH="$1"
endif

if ("$PSTL_TARGET_ARCH" != "") then
    if ("$PSTL_TARGET_ARCH" != "ia32" && "$PSTL_TARGET_ARCH" != "intel64") then
        echo "ERROR: Unknown switch '$PSTL_TARGET_ARCH'. Accepted values: ia32, intel64"
        set PSTL_TARGET_ARCH=""
        exit 1
    endif
else
    echo "ERROR: Architecture is not defined. Accepted values: ia32, intel64"
    exit 1
endif


# Arg2 represents PSTLROOT detection method. Its possible value is 'auto_pstlroot'. In which case
# the environment variable PSTLROOT is detected automatically by using the script directory path.
if ("$2" == "auto_pstlroot") then
    set sourced=($_)
    if ("$sourced" != '') then # if the script was sourced
        set script_name=`readlink -f $sourced[2]`
    else # if the script was run => "$_" is empty
        set script_name=`readlink -f $0`
    endif
    set script_dir=`dirname $script_name`
    setenv PSTLROOT "$script_dir/.."
else
    setenv PSTLROOT "SUBSTITUTE_INSTALL_DIR_HERE"
endif

if ( -e $PSTLROOT/../tbb/bin/tbbvars.csh ) then
   source $PSTLROOT/../tbb/bin/tbbvars.csh $PSTL_TARGET_ARCH;
endif

if (! $?CPATH) then
    setenv CPATH "${PSTLROOT}/include"
else
    setenv CPATH "${PSTLROOT}/include:$CPATH"
endif
