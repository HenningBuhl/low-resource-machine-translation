#!/bin/bash

# Function to check if a list contains an item.
function contains {
  local list="$1"
  local item="$2"
  if [[ $list =~ (^|[[:space:]])"$item"($|[[:space:]]) ]] ; then
    result=0
  else
    result=1
  fi
  return $result
}

# Special arguments that only affect the bash script.
SPECIAL_ARGS="EXPERIMENT CONDA_PATH CONDA_ENV SKIP_CONVERT"
EXPERIMENT="Baseline"
CONDA_PATH=""
CONDA_ENV=""
SKIP_CONVERT=false

# Save script name without .sh file ending.
SCRIPT_NAME=`basename "$0"`
SCRIPT_NAME=${SCRIPT_NAME%".sh"}

# Create string containing arguments for python call.
PARGS=""

# Iterate over all arguments.
for ARGUMENT in "$@"
do
    # Extract argument key.
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    
    # Do not add arguments to the python arguments that are only meant for the sh-script.
    if `contains "$SPECIAL_ARGS" "$KEY"` ; then
        if [[ $KEY == "EXPERIMENT" ]]; then
            EXPERIMENT=$(echo $ARGUMENT | cut -f2 -d=)
        elif [[ $KEY == "CONDA_PATH" ]]; then
            CONDA_PATH=$(echo $ARGUMENT | cut -f2 -d=)
        elif [[ $KEY == "CONDA_ENV" ]]; then
            CONDA_ENV=$(echo $ARGUMENT | cut -f2 -d=)
        elif [[ $KEY == "SKIP_CONVERT" ]]; then
            SKIP_CONVERT=true
        fi
        continue
    fi
    
    # Convert key from pascal case to spinal case.
    KEY=$(echo $KEY | sed -r 's/([A-Z])/-\L\1/g' | sed 's/^-//')
    
    # Append key to python arguments.
    PARGS="$PARGS --$KEY"
    
    # Extract argument value (if present).
    if [[ $ARGUMENT == *"="* ]]; then
        VALUE=$(echo $ARGUMENT | cut -f2 -d=)
        
        # Replace ',' with ' ' in value (for arguments that are lists).
        VALUE=$(echo $VALUE | sed -r 's/,/ /g')
        
        # Append value to python arguments.
        PARGS="$PARGS $VALUE"
    fi
done

# Activate conda with env if requested.
if [[ $CONDA_ENV != "" ]]; then
    # Use the default minconda path if CONDA_PATH is an empty string.
    if [[ $CONDA_PATH == "" ]]; then
        CONDA_PATH="~/miniconda3/etc/profile.d/conda.sh"
    fi
    # Activate conda env.
    source "$CONDA_PATH"
    conda activate "$CONDA_ENV"
fi

# Convert notebook to python file if not skipped.
if [ $SKIP_CONVERT == false ]; then
    jupyter nbconvert --to python "$EXPERIMENT".ipynb
fi

# Call pyhton with the arguments.
#echo "$PARGS"
python "$EXPERIMENT".py ${PARGS}
