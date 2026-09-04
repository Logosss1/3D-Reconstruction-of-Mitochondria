# Data Sources and Licensing Boundaries

The experiments use volumetric electron microscopy data and mitochondrial
labels from the OpenOrganelle/Janelia ecosystem, accessed through
OpenOrganelle-compatible layouts or exported crop Zarr files.

## Referenced datasets

- **JRC HeLa-2 / HeLa2**: Janelia data record, current version DOI
  [10.25378/janelia.13114211.v2](https://doi.org/10.25378/janelia.13114211.v2).
  The unversioned record identifier is `10.25378/janelia.13114211`.
- **JRC HeLa-3 / HeLa3**: Janelia data record, current version DOI
  [10.25378/janelia.13114244.v2](https://doi.org/10.25378/janelia.13114244.v2).
  The unversioned record identifier is `10.25378/janelia.13114244`.

The records currently identify **Version 2**, released on December 12, 2024.
Use the versioned DOI when you need the exact dataset state used for a
reproducible run; use the unversioned identifier when referring to the record
as a whole.
- **OpenOrganelle project**: dataset access, metadata, and attribution should
  follow the terms published by the OpenOrganelle/Janelia source record used
  for a particular run.

## Local data and derived outputs

Files under `data/`, `checkpoints/`, `reconstruction/`, and result folders may
be raw, converted, derived, or generated artifacts. Their presence in this
repository does not change the terms of the upstream data source. Before
redistributing a raw volume, label, checkpoint, or derived crop, verify the
applicable source record and permissions.

## Processing summary

The documented workflow loads source volumes or crop Zarr files, constructs
training crops, trains an occupancy model, evaluates dense occupancy fields,
extracts meshes with Marching Cubes, and computes crop-level validation
metrics. The exact crop selection and post-processing parameters should be
recorded alongside any published result.
