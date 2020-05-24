#!/bin/bash

export MODULEPATH=/global/software/sl-7.x86_64/modfiles/langs:/global/software/sl-7.x86_64/modfiles/tools:/global/software/sl-7.x86_64/modfiles/apps:/global/home/groups/consultsw/sl-7.x86_64/modfiles:/clusterfs/vector/home/groups/software/sl-7.x86_64/modfiles

export DISPLAY=10.0.0.25$DISPLAY

source /etc/profile.d/modules.sh

cd /global/software/sl-7.x86_64/modules/tools/matlab/r2019b/bin/glnxa64
       ./SupportSoftwareInstaller -downloadfolder /global/home/users/seenum/MATLAB-AddOns/R2019b -inputfile /global/home/users/seenum/MATLAB-AddOns/R2019b/ssi_input.txt


module load matlab/r2019b

matlab -desktop
