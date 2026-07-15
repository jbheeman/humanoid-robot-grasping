#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-$HOME/g1-bunny-vla}"
mkdir -p "$project_root" "$project_root/data/hdf5" "$project_root/data/rlds" "$project_root/logs"

if [[ ! -d "$project_root/unifolm-vla/.git" ]]; then
  git clone https://github.com/unitreerobotics/unifolm-vla.git "$project_root/unifolm-vla"
fi

if [[ ! -d "$project_root/unitree_sim_isaaclab/.git" ]]; then
  git clone --recurse-submodules https://github.com/unitreerobotics/unitree_sim_isaaclab.git \
    "$project_root/unitree_sim_isaaclab"
fi

if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  conda_bin="$HOME/miniconda3/bin/conda"
elif [[ -x "$HOME/miniconda/bin/conda" ]]; then
  conda_bin="$HOME/miniconda/bin/conda"
else
  echo "Miniconda was not found under ~/miniconda3 or ~/miniconda" >&2
  exit 2
fi

if ! "$conda_bin" env list | awk '{print $1}' | grep -qx g1-bunny-data; then
  "$conda_bin" create -y -n g1-bunny-data python=3.10
fi

"$conda_bin" run -n g1-bunny-data python -m pip install numpy h5py tensorflow-datasets
echo "remote project initialized at $project_root"
echo "Isaac-specific packages must be run with Isaac Sim's bundled Python, not this conversion environment."
