# Data & cohort

## 1. ADNI data model

The project uses ADNI, a longitudinal multi-centre observational dataset that includes clinical, cognitive, imaging, biofluid, and genetic information.

The core data representation is a participant × visit table with longitudinal alignment on participant and visit/time information.

| RID | VISCODE | EXAMDATE | DX | MMSE | ADAS13 | CDRSB | MRI | PET | CSF | APOE |
|---|---|---|---|---|---|---|---|---|---|---|
| 001 | bl | … | CN | 29 | 8 | 0 | ✓ | – | – | 0 |
| 001 | m12 | … | MCI | 26 | 14 | 1.5 | ✓ | ✓ | – | 0 |
| 001 | m24 | … | MCI | 24 | 19 | 3 | ✓ | ✓ | ✓ | 0 |
| 001 | m36 | … | AD | 20 | 28 | 6 | ✓ | ✓ | ✓ | 0 |

## 2. Access and provenance

ADNI files are downloaded manually and stored under `data/raw/`.

Required provenance includes:

- file name
- ADNI phase / release
- download date
- processing version
- dictionary version
- checksum or source metadata

## 3. Modality groups

| Modality | Representative information |
|---|---|
| Clinical | diagnosis, comorbidities, function, assessments |
| Cognitive | ADAS-Cog, MMSE, CDR, MoCA, FAQ |
| MRI | processed ROI measures |
| PET | FDG, amyloid, tau |
| Biofluid | CSF / blood biomarkers |
| Genetic | APOE ε4 allele count |

## 4. Data rules

1. Never edit raw ADNI files directly.
2. Keep all processing reproducible from code + configuration.
3. Preserve provenance and dictionary metadata.
4. Keep sensitive identifiers only in the approved research environment.
5. Never commit raw ADNI data to public version control.

## 5. Preprocessing expectations

- standardization using training-set statistics only
- MRI ROI normalization by intracranial volume where appropriate
- PET SUVR normalization using reference-region uptake
- APOE encoded as number of ε4 alleles
- feature selection fit only within training folds

## 6. Cohort design and leakage prevention

Participant-level splits are required. Repeated visits from the same participant must stay within the same fold.

This protects against leakage between train, validation, and test partitions.

## Related pages

- [Overview](../overview/README.md)
- [Methods](../methods/README.md)
- [Evaluation & trustworthiness](../evaluation/README.md)
- [Implementation & roadmap](../project/README.md)
