# R2S Fusion Standards
Standardization of fusion R2s workflows for better communication and validation.

## Format

For each `problem`, the root directory contains geometry files (`.cub, mcnp files`) and material data (`mcnp file`)
to reproduce the problem. There are folders, named by the R2S workflow (e.g., `orcs`, `shift-pyne`) that contain
the following files:
1. MG neutron mesh tally
2. Gamma emission source distribution
3. MG gamma mesh tally


## Folder Contents


### scripts

Contains scripts for reading and writing r2s data
1. `mesh_data.py`: python class for all mesh data, contains universal mesh data properties and functions
2. `neutron_mesh_tally.py`: uses `mesh_data` for neutron mesh tallies. Reading and writing functions for specific file formats.
3. `gamma_source_mesh.py`: uses `mesh_data` for gamma emission mesh data. Reading and writing functions for specific file formats.


### problems

1. `iter_sddr_benchmark`

![iter_sddr_benchmark](https://github.com/jbae11/r2s_standards/problems/iter_sddr_benchmark/imgs/iter_sddr_comp_benchmark.png)
