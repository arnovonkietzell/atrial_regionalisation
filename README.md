# Atrial Regionalisation

Code to support automatic regionalisation of the left and right atrium into
the 15 segments of the EHRA/EACVI bi-atrial model (Althoff et al., Europace
2025;27:euaf134), from a set of interactively placed anatomical landmarks.
This repository contains both a source library of utilities `ar_utils` and
a collection `wips` of WIP modules designed to be run in the EP Workbench
software.

## Pre-requisites

- **Conda** (or your preferred Python environment) is installed.
- The [OpenEP library](https://github.com/ecci-cvs/openep-py) and its
  dependencies are installed in your environment:

```bash
conda create -n ar python=3.10 pip
conda activate ar
git clone https://github.com/ecci-cvs/openep-py
cd openep-py
python -m pip install -e .
```
- (Recommended) EP Workbench, minimal version `v1.1.0-beta.1-260505`. Download instructions available [here](https://openep.discourse.group/t/downloading-ep-workbench-beta-for-academic-use/149).

## How to run this WIP

1. Ensure all pre-requisites are installed on your computer (see above).

2. Clone this repository and install requirements:

```bash
conda activate ar
git clone https://github.com/arnovonkietzell/atrial_regionalisation
cd atrial_regionalisation
pip install -r requirements.txt
which python #save this env python path for later
```

3. Running the WIP through EP Workbench:

- Open EP Workbench and navigate to Work-in-Progress > Marketplace.
- Select "Atrial Regionalisation" and press "clone".
- On the WIP Bar, select Info > Edit.
- Set Interpreter as the path to your python environment (from step 2), and Root Dir as the path to `atrial_regionalisation` (this repository). Save these settings and close the WIP editor window.
- You are now ready to run the WIP - see `wips/atrial_regionalisation/README.md` for details.

4. Running the WIP outside EP Workbench:

To run in `debug` mode, open the WIP's
`main.py` and edit `mesh_path` / `root_dir` / `chamber` to your preferred
input mesh and output directory, then:

```bash
conda activate ar
cd atrial_regionalisation
python -m wips.atrial_regionalisation.main
```

## Overview

- `atrial_regionalisation`: places the 15 EHRA/EACVI bi-atrial model
  landmarks on an LA or RA mesh and computes the corresponding 15-segment
  `cell_region` labelling. See `wips/atrial_regionalisation/README.md` for
  details.
