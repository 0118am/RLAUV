#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
target_dir="${script_dir}/.runtime/geometry/site-packages"

if PYTHONPATH="${target_dir}${PYTHONPATH:+:${PYTHONPATH}}" /usr/bin/python3 -c \
  'import numpy, scipy, skimage, vtk; assert numpy.__version__ == "1.26.4"; assert scipy.__version__ == "1.15.0"; assert skimage.__version__ == "0.25.2"; assert vtk.vtkVersion.GetVTKVersion() == "9.5.2"' \
  >/dev/null 2>&1; then
    printf 'Geometry runtime is ready: %s\n' "${target_dir}"
    exit 0
fi

mkdir -p "${target_dir}"
/usr/bin/python3 -m pip install \
  --disable-pip-version-check \
  --upgrade \
  --target "${target_dir}" \
  numpy==1.26.4 \
  scipy==1.15.0 \
  scikit-image==0.25.2 \
  vtk==9.5.2 \
  networkx==3.4.2 \
  pillow==12.3.0 \
  imageio==2.37.4 \
  lazy-loader==0.5 \
  tifffile==2025.5.10 \
  packaging==26.3 \
  matplotlib==3.10.9 \
  fonttools==4.63.0 \
  cycler==0.12.1 \
  contourpy==1.3.2 \
  kiwisolver==1.5.0 \
  pyparsing==3.3.2 \
  python-dateutil==2.9.0.post0 \
  six==1.17.0

PYTHONPATH="${target_dir}${PYTHONPATH:+:${PYTHONPATH}}" /usr/bin/python3 -c \
  'import numpy, scipy, skimage, vtk; assert numpy.__version__ == "1.26.4"; assert scipy.__version__ == "1.15.0"; assert skimage.__version__ == "0.25.2"; assert vtk.vtkVersion.GetVTKVersion() == "9.5.2"'
