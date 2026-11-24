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

!!! warning "Being verified"
    This section will list the exact dataset, version, DOI, license text, number of clips per class and how we split it, once `model/download_data.py` has checked the source (build step 2).

## Why a "background" class matters

A classifier always picks *one* of its labels. Without a "no primate" option, it would call wind, rain, insects and birds "gibbon" with high confidence. The background class gives the model a correct answer for "nothing interesting here", which is what a robot hears most of the time.
