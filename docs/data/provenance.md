# PLAsTiCC Dataset Provenance and License

## Source

**Dataset:** PLAsTiCC (Photometric LSST Astronomical Time-Series Classification Challenge)  
**Release:** Public unblinded test-population release  
**Zenodo record:** [https://zenodo.org/records/2539456](https://zenodo.org/records/2539456)  
**File used:** `plasticc_test_metadata.csv.gz`  
**License:** [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/)

## Description

PLAsTiCC is an LSST-like photometric transient classification challenge.
Its test population was generated using the LSST Operations Simulator (OpSim) and
SNANA to simulate multi-band (ugrizy) light curves of diverse astronomical transient
classes at realistic survey cadence, depth, and noise levels.

The test metadata table (`plasticc_test_metadata.csv.gz`) contains one row per
simulated object. It is the basis for the AEGIS **TRUE population**: the full
simulated survey population restricted to the three pre-registered study classes
(ADR 002). It represents what would be detected without any spectroscopic
follow-up selection.

The released training set was produced by a separate simulation process and is
**not** used in AEGIS population construction (see ADR 004 for the reasoning).

## Literature References

- **PLAsTiCC data-set paper:** Kessler et al. (2019), arXiv:1903.11756  
  <https://arxiv.org/abs/1903.11756>
- **PLAsTiCC model reveal:** LSST Dark Energy Science Collaboration (2019)  
  <https://plasticc.org/wp-content/uploads/2019/01/plasticc_modelreveal_2versions.pdf>

## Access and Checksums

The file is downloaded by the pipeline script (`scripts/ingest_population.py --stage raw`).
The raw-stage manifest (`data/raw/manifest.json`) records:

- `sha256`: SHA-256 checksum of the downloaded `.csv.gz` file
- `source_url`: Zenodo direct download URL
- `license`: CC-BY-4.0
- `dataset`: `plasticc_test_metadata`

The manifest is written after every run. Checksums can be compared to reproduce
the exact provenance chain from source to processed populations.

## Acknowledged Limitations

1. PLAsTiCC simulations cover a model family of kilonova light curves (class 64).
   Results may not generalize to kilonova populations outside that family.
2. The test-population class distribution reflects the PLAsTiCC simulation design,
   not the true astrophysical rate density weighted by LSST's detection efficiency.
3. Host galaxy fields (`hostgal_photoz`, `hostgal_photoz_err`) are simulation
   outputs, not measured spectroscopic redshifts.

## Citation

If you use AEGIS results derived from this data, cite the PLAsTiCC data-set paper
(Kessler et al. 2019, arXiv:1903.11756) and the Zenodo record (doi:10.5281/zenodo.2539456).
