# CReTA
CReTA (**C**reak **Re**cognition via **T**onal **A**nalysis) is a creaky voice detection algorithm presented as a poster in the 2025 *Fonetiikan päivät* conference in Turku, Finland (more information [here](https://researchportal.helsinki.fi/fi/publications/presenting-an-alternative-creak-detection-algorithm)). At that time, the algorithm didn't yet have a name.

This is based on comparing the output of two different pitch (fundamental frequency, f0) detection algorithms. Since octave jumps in automatic pitch detection are often caused by creaky voice (e.g. [Watkins et al. 2024](https://www.isca-archive.org/interspeech_2024/watkins24_interspeech.html)), this means that by locating the octave jumps we could identify where creaky has a high probability of existing. If two algorithms, one that is robust and makes few octave jumps, and one less robust one which makes many octave jumps, disagree with each other by a large enough margin, those areas get labeled creaky. Unless you modify the code yourself, this algorithm uses PitchSqueezer as the robust algorithm, and cc (cross-correlation) as the less robust one. There is also an option (by changing the line `a3 = None` near the end of `main.py` to eg. `a3 = 'shs'`) to also include a third algorithm with the aim of preventing false positives: sections that are completely voiceless according to a3 will not be labeled creaky in this case. However, based on the testing we did before the conference, creak detection worked better without the third algorithm, thus a3 is assigned None in this code.

Since PitchSqueezer, which this algorithm relies on, is not deterministic, we used `numpy.random.seed` together with `os.environ` commands to make this as deterministic as possible. The results should be identical when run on one computer, but accross computers they may vary slightly.

The parameters defined at the end of the `main.py` module (from the line `floors = [50, 60, 60]` to `max_candidatess = [15, 15, 15]`) were the ones that on average worked the best in the datasets we tested it on (subsets of the [PROSO-ASD](https://blogs.helsinki.fi/asd-prosody-research/) project datasets). The parameters here had the greatest average and accuracy and the lowest accuracy standard deviation per dataset, compared to all other sets of parameters we tested it on. The datasets were a collection of autistic French speech from Geneva, a collection of autistic French speech collected in Orbe, a collection of neurotypical French speech from Lausanne, Finnish autistic speech, Finnish neurotypical speech, Slovak autistic speech, and Slovak neurotypical speech (small subsets of each), where the Slovak speech came from the Slovak Autistic and Non-Autistic Child Speech Corpus ([SANACS](https://catalog.elra.info/en-us/repository/browse/ELRA-S0491/)). Before the Fonetiikan päivät 2025 conference, we had tested this algorithm against the [Covarep](https://sites.bu.edu/stepplab/research/automated-creak-detection/) creak detection algorithm, and for the French speech as well as the Slovak autistic speech, CReTA worked better, but Covarep worked better for Finnish and neurotypical Slovak.

A great deal of this code was written by or in collaboration with ChatGPT.

### How to run

To run this, you need to:
1. Download both `main.py` and `tgread.py` and place them in the same folder.
2. Install Python. We have tested this on [Python 3.9.9](https://www.python.org/downloads/release/python-399/) on Windows 10 and 11, but it might run on other versions too (definitely not < 3.6 though) and probably also on other operating systems.
3. Install [PitchSqueezer](https://github.com/asuni/PitchSqueezer) and its dependencies.
4. Install [Parselmouth](https://pypi.org/project/praat-parselmouth/) and its dependencies.
5. Run `main.py`. It asks which folder to analyze. Type in (or copy and paste) the folder whose WAV files you wish to analyze, and the script creates `TextGrid` files (understood by Praat) with a single tier, where segments detected as creaky are labelled "n". It also creates `.f0.txt` files, containing list of the file's fundamental frequencies, as well as two `mrawf0` files, where they are mapped to the correct times. The file ending in `_withcreaks.mrawf0` has creaky points mapped to F0 = 1 (Hz). However, the parameters that work best for creak detection might not be the best available ones for F0 detection as its own task, meaning that you most likely want to ignore the non-`TextGrid` files created by CReTA (and PitchSqueezer when you're running CReTA).

### If you edit this code

If you edit this code, do not set more than one of the set {a1, a2, a3} to be 'pitchsqueezer'.

You may find some of these functions useful in other projects too, such as the helper functions in `tgread.py` and the `to_pitch_praat` function in `main.py`, which converts a sound into a `parselmouth.Pitch` object.

### Citation:
Haakana, V. & Jokinen, R. (2025, April 24–25). “Presenting an Alternative Creak Detection Algorithm” [Poster presentation], _XXXVII Fonetiikan päivät_, Turku.
