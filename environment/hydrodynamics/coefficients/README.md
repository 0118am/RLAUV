# Hydrodynamic coefficient sets

Files here are runtime inputs consumed directly by Isaac. The current OpenFOAM
file contains the three selected full-response matrices from campaign
`ffa066cfd002cdbb8c57b674f07edf25da6c603c499ab0992625df24df32162a`.
Its allowed off-diagonal entries are active runtime coefficients; no separate
diagonal fallback exists. CFD acceptance or provenance status does not control
whether the simulator loads the selected profile.
Raw OpenFOAM and PMM outputs remain in their source directories; PMM results
are not a production coefficient source.
