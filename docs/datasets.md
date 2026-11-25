# Datasets

We only use audio we are allowed to use. Before any dataset goes into training, we check its license and its download link from the source page itself — not from memory or a blog post. Raw audio is **never committed** to this repository; `model/download_data.py` downloads it.

## Candidates we reviewed

| Dataset | What's in it | Status |
|---|---|---|
| **ComParE 2021 Primates** (Zwerts et al., 2021) | Chimpanzee, mandrill, red-capped mangabey, collared mangabey + background, from a sanctuary in Cameroon | Not used — we found no open download; ComParE challenge data is normally shared under a signed agreement. |
| **MarmAudio** (Lamothe et al., 2025) | ~215k labelled marmoset calls (phee, trill, twitter, …) | Not used — about 60 GB, far too large for a laptop project. |
| **InfantMarmosetsVox** (Sarkar & Magimai-Doss, 2023) | Infant marmoset call types and caller IDs | Backup option — custom license; no background class. |
| **Borneo loud calls** (Clink et al., Zenodo) | Pre-cut clips: female gibbon, hornbills, great argus pheasant, **noise** | First choice — license and file list being verified. |

## The dataset we use

> Clink, D. J. & Hamid Ahmad, A. (2024). *A labelled dataset of the loud calls of four vertebrates collected using passive acoustic monitoring in Malaysian Borneo.* Zenodo. [doi:10.5281/zenodo.14213067](https://doi.org/10.5281/zenodo.14213067). License: **CC-BY-4.0**.

- **Where:** Danum Valley Conservation Area, Sabah, Malaysia (2018), with Swift autonomous recorders strapped to trees. A second, smaller set comes from Maliau Basin (2019).
- **Format:** 16 kHz mono WAV clips, 6–20 seconds each, cut out of 2-hour recordings by an energy detector and then labelled by an expert.
- **License check:** `model/download_data.py` reads the license from the Zenodo API and stops unless it is an approved open license. It also checks every file's MD5 checksum.
- **The primate:** the **Northern grey gibbon** (*Hylobates funereus*). Gibbons are small apes, not monkeys. Their loud, musical "great calls" carry ~400 m through the forest.
- **Other classes:** three birds that call in the same pitch range (great argus pheasant, helmeted hornbill, rhinoceros hornbill), plus a catch-all **noise** class that we use as "background".

### Clips per split

We split by **recording**: all clips cut from the same 2-hour recording go into the same split (see [How It Works](how_it_works.md#train-validation-test-split)).

| Split | Gibbon | Great argus | Helmeted hornbill | Rhinoceros hornbill | Background | Total |
|---|---|---|---|---|---|---|
| Train | 310 | 62 | 50 | 18 | 575 | 1,015 |
| Validation | 61 | 13 | 10 | 4 | 115 | 203 |
| Test | 62 | 12 | 10 | 4 | 116 | 204 |
| Maliau (extra test) | 110 | — | — | 4 | 95 | 209 |

We download `trainingclips.zip` (353 MB) and `testclipsmaliau.zip` (41 MB). The record also has larger test sets that we don't use yet.

!!! warning "Why only one primate?"
    We looked hard for an openly licensed dataset with several *monkey* species and a background class. The best-known one (ComParE 2021 Primates: chimpanzee, mandrill, two mangabeys) had no open download we could find. If your team gets access to more primate audio under an open license, add a folder per species and re-run `python model/train.py`: the code doesn't need to change beyond `model/config.py`.

## Why a "background" class matters

A classifier always picks *one* of its labels. Without a "no primate" option, it would call wind, rain, insects and birds "gibbon" with high confidence. The background class gives the model a correct answer for "nothing interesting here", which is what a robot hears most of the time.
