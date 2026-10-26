# Dataset attribution and reuse

The MIT license applies to repository code, not third-party ECG data.

SHDB-AF version 1.0.1: Tsutsui, K., Biton Brimer, S., and Behar, J. (2025).
[PhysioNet resource](https://physionet.org/content/shdb-af/1.0.1/),
[dataset DOI 10.13026/n6yq-fq90](https://doi.org/10.13026/n6yq-fq90),
[associated publication DOI 10.1038/s41597-025-04777-4](https://doi.org/10.1038/s41597-025-04777-4).
Published under [Open Data Commons Attribution License v1.0](https://physionet.org/files/shdb-af/1.0.1/LICENSE.txt).
Retain this attribution and the dataset license in downstream data/model packages and presentations.
The downloader saves LICENSE.txt and SHA256SUMS.txt alongside local data.

The repository's demo generator manufactures ECGs, pseudo clinical profiles and precursor patterns.
These fixtures are code-test examples, not de-identified patients or evidence of AF forecasting performance.
Do not publish them as Indian-population validation.

The optional [PAF Prediction Challenge](https://physionet.org/content/afpdb/1.0.0/) is a separate ECG-only benchmark,
not external validation of the fused continuous SHDB-AF forecast; no unrelated clinical records may be joined.


## Primary forecasting cohort: IRIDIA-AF

Gilon C, Grégoire JM, Mathieu M, Carlier S, Bersini H (2023).
[IRIDIA-AF v1.0.1, Zenodo DOI 10.5281/zenodo.8405941](https://zenodo.org/records/8405941),
[associated Scientific Data publication](https://doi.org/10.1038/s41597-023-02621-1),
[authors' ingestion repository](https://github.com/cedricgilon/iridia-af).
The saved official release metadata specifies **CC BY 4.0**;
[license terms](https://creativecommons.org/licenses/by/4.0/).
Retain author attribution, source/version, license and descriptions of transformations when redistributing data derivatives.
The software license does not replace these dataset terms.

The user authorized acquisition of the full public cohort for this investigation, with a 25 GB peak local dataset budget.
Before extraction, the archive directory established 10,532,846,900 compressed bytes and 10,954,900,447 uncompressed bytes:
21,487,747,347 bytes for archive plus extraction, within that budget. All published records were selected before reading
annotations; there was no outcome-based acquisition subset. The complete archive was verified against published MD5
`81d58d21c9835a601cd9eb6c68bf9feb`; metadata against `f0102de28265ed6459956ac6cd397464`.
Extraction checked ZIP member CRCs and recorded local SHA256 hashes, then removed the redundant archive.
These checks establish file identity and integrity, not clinical measurement validity.

Local provenance: `data/iridia-af/release-metadata.json`, `source.json`, `archive-inventory.json` and `file-checksums.json`;
summary: [cohort audit](reports/ml-investigation/cohort-audit.json).
There are 167 recordings from 152 explicit patients, with 289 daily ECG files. Reads preserve the concatenated sample timeline.
The first 30 seconds of each daily file are excluded from signal inputs and marked unknown in labels, following the authors'
`clean_front` convention. Age and sex are the available paired clinical covariates. Vendor automated RR files are preserved
as source material and are not training/runtime inputs or reviewed beat ground truth. These are public research recordings;
this project has not acquired new patients or established prospective clinical consent or approval.

## Reserved external cohort: LTAF

[Long Term AF Database v1.0.0](https://physionet.org/content/ltafdb/1.0.0/),
[DOI 10.13026/C2QG6Q](https://doi.org/10.13026/C2QG6Q),
[Open Data Commons Attribution License v1.0](https://physionet.org/content/ltafdb/view-license/1.0.0/).
Original publication: Petrutiu S, Sahakian AV, Swiryn S (2007),
“Abrupt changes in fibrillatory wave characteristics at the termination of paroxysmal atrial fibrillation in humans,”
Europace 9:466–470. Acknowledge the reviewed `.atr` annotation creators, MEDICALgorithmics Ltd (Warsaw, Poland),
coordinated by Michal Tadeusiak. `.qrs` detections are automated and are not the reviewed rhythm endpoint.

All 84 official record headers and reviewed `.atr` files were acquired and checked against published SHA256SUMS.
`data/ltaf/source.json`, `SHA256SUMS.txt` and `LICENSE-and-description.html` retain provenance.
Signals are reserved for evaluation after finalist/policy freeze. Recording IDs alone do not prove unique subjects or
cross-dataset subject separation; the project does not claim that identity audit has been resolved.

## Beat measurement and noise benchmarks

[MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/),
[DOI 10.13026/C2F305](https://doi.org/10.13026/C2F305),
[Open Data Commons Attribution License v1.0](https://physionet.org/content/mitdb/view-license/1.0.0/).
Cite Moody GB, Mark RG (2001), “The impact of the MIT-BIH Arrhythmia Database,” IEEE Engineering in Medicine and Biology
20(3):45–50. The investigation acquired all 48 complete recordings and reviewed beat annotations, expanding the earlier
600-second checks on records 100 and 201. Records 201 and 202 belong to one person and must remain linked.

[MIT-BIH Noise Stress Test Database v1.0.0](https://physionet.org/content/nstdb/1.0.0/),
[DOI 10.13026/C2HS3T](https://doi.org/10.13026/C2HS3T),
[Open Data Commons Attribution License v1.0](https://physionet.org/content/nstdb/view-license/1.0.0/).
Cite Moody GB, Muldrow WE, Mark RG (1984), “A noise stress test for arrhythmia detectors,” Computers in Cardiology
11:381–384. The 12 noisy recordings derive from MIT-BIH records 118/119; three noise-only records complete the release.
Noise variants are repeated measurements of the source ECGs, not independent people.

Both measurement cohorts were obtained from official PhysioNet files. Their `data/measurement/{mitdb,nstdb}/acquisition.json`
files record local SHA256 hashes and license links. **These hashes are local reproducibility checks, not independent
published-checksum verification.** The comparisons measure beat detection and robustness; they do not validate AF forecasting
or provide paired clinical metadata. Include the standard PhysioNet citation requested by the source resource when presenting
these databases; see [PhysioNet's current citation](https://doi.org/10.1038/s44360-026-00096-z).

Raw datasets remain outside Git and application images. User authorization to acquire public data for local research is not
new patient consent, a clinical-readiness claim, or permission to remove source attribution. The authorized GPU plan keeps
runs private; authentication and quota must be verified before dispatch. No Kaggle upload or job occurred during the
read-only connector investigation, and no token is included in project configuration or provenance reports.
