#!/usr/bin/env bash
# Convert the PhysioNet credentialed zips (MedNLI, MIMIC-III trial questions, MIMIC-IV name de-id)
# into prepared/. Run by the licensee on Kelvin2 only. Prints counts, never text.
#   bash scripts/build_clinical.sh $SCRATCH/physionet_zips
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: bash scripts/build_clinical.sh DIR_WITH_PHYSIONET_ZIPS" >&2; exit 2; }
source hpc/env.sh
umask 077
python data_build/build_clinical.py --src "$1" --out prepared
