import os
# Ensuring determinism (trying at least)
# Core numerical libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# Just in case
os.environ["NUMBA_NUM_THREADS"] = "1"
# If using TensorFlow (optional)
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
import math
import numpy as np
#import subprocess
import wave
from fractions import Fraction
import warnings
import parselmouth
from pitch_squeezer import track_pitch, f0_cwt
from fractions import Fraction
from tgread import read_tg

if __name__ == '__main__':

    output_folder = input('Analyze which folder? ')
    

def avaa(x):
    with open(x, 'rt', encoding='utf-8') as k:
        b = k.read()
        return b[b.startswith('\ufeff'):]

def to_pitch_praat(sound, resample=False, pitch_floor=75.0, pitch_ceiling=600.0, time_step=None, method='cc',
                   octave_jump_cost=0.35, voicing_threshold=0.45, voiced_unvoiced_cost=0.14, 
                   very_accurate=False, silence_threshold=0.03, max_candidates=15, 
                   mode='return'):
    """
    Extract pitch using Praat's algorithms with various parameters.

    Parameters:
    - sound: path to the sound file or a parselmouth.Sound object.
    - pitch_floor: Minimum pitch frequency (Hz).
    - pitch_ceiling: Maximum pitch frequency (Hz).
    - time_step: Time step for pitch analysis.
    - method: Algorithm to use ('ac', 'cc', or 'shs').
    - octave_jump_cost: Cost of octave jumps.
    - voicing_threshold: Threshold for voicing decisions.
    - voiced_unvoiced_cost: Cost of voiced/unvoiced transitions.
    - very_accurate: Use very accurate mode (boolean).
    - silence_threshold: Threshold for silence detection.
    - max_candidates: Maximum number of pitch candidates.
    - mode: 'return' to return the Pitch object, or 'save <path>' to save as text.

    Returns:
    - parselmouth.Pitch object if mode is 'return'.
    """
    # Load the audio file
    if isinstance(sound, str):
        sound = parselmouth.Sound(sound)

    if resample:
        resampled_sound = sound.resample(resample)
    else:
        resampled_sound = sound

    # Handle method-specific parameter orders
    if method == 'ac':
        pitch = parselmouth.praat.call(
            resampled_sound, "To Pitch (ac)", time_step or 0, pitch_floor, max_candidates,
            "yes" if very_accurate else "no", silence_threshold, voicing_threshold, 0.01,
            octave_jump_cost, voiced_unvoiced_cost, pitch_ceiling
        )
    elif method == 'cc':
        pitch = parselmouth.praat.call(
            resampled_sound, "To Pitch (cc)", time_step or 0, pitch_floor, max_candidates,
            "yes" if very_accurate else "no", silence_threshold, voicing_threshold, 0.01,
            octave_jump_cost, voiced_unvoiced_cost, pitch_ceiling
        )
    elif method == 'shs':
        pitch = parselmouth.praat.call(
            resampled_sound, "To Pitch (shs)", time_step or 0.01, pitch_floor, max_candidates,
            1250, 15, 0.84, pitch_ceiling, 48
        )
    else:
        raise ValueError(f"Unsupported method '{method}'. Use 'ac', 'cc', or 'shs'.")

    # Return or save the pitch object
    if mode == 'return':
        return pitch
    elif mode.startswith('save '):
        path = mode[5:]
        pitch.save(path, parselmouth.Data.FileFormat.TEXT)
    else:
        raise ValueError("Invalid mode. Use 'return' or 'save <path>'.")

def to_matrix(pitch, mode='return'):
    """mode must be either 'return' or 'save ' + path"""
    matrix = pitch.to_matrix()

    if mode == 'return':
        return matrix
    else:
        assert mode.startswith('save ')
        matrix.save(mode[5:], parselmouth.Data.FileFormat.TEXT)

def pitch_to_tuple(pitch):
    if type(pitch) == str:
        pitch = parselmouth.read(pitch)

    answer = []

    for n, t in enumerate(zip(pitch.xs(), pitch.selected_array['frequency'])):
        answer.append((n+1, t[0], t[1]))

    return answer

def mrawf0_to_tuple(file_path): # written by chatgpt
    """
    Reads a text file and outputs a list of tuples with interpolated missing points.

    Parameters:
    - file_path: Path to the text file to read.

    Returns:
    - List of tuples in the format [(point, time, F0), ...].
    """
    result = []

    # Read the file and parse the lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skip the header and process the rest
    data = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 3:
            point, time, f0 = int(parts[0]), float(parts[1]), float(parts[2])
            data.append((point, time, f0))

    # Fill in missing points
    for i in range(len(data) - 1):
        current_point, current_time, current_f0 = data[i]
        next_point, next_time, next_f0 = data[i + 1]

        # Add the current point to the result
        result.append((current_point, current_time, current_f0))

        # Check for missing points
        if next_point > current_point + 1:
            for missing_point in range(current_point + 1, next_point):
                # Interpolate time linearly
                interpolated_time = current_time + (next_time - current_time) * (missing_point - current_point) / (next_point - current_point)
                # Set F0 to 0.0
                result.append((missing_point, interpolated_time, 0.0))

    # Add the last point
    result.append(data[-1])

    return result

#ChatGPT edit
def matrix_to_tuple(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    time_step = None
    first_time = None
    frequencies = []
    points = []

    for line in lines:
        if line.startswith("dx = "):
            time_step = float(line.strip().split('=')[-1].strip())
        elif line.startswith("x1 = "):
            first_time = float(line.strip().split('=')[-1].strip())
        elif line.strip().startswith("z [1] ["):
            frequency = line.split('=')[-1].strip()

            point_idx = line.split('] [')[1].split(']')[0]
            time = first_time + int(point_idx) * time_step

            frequencies.append(frequency)
            points.append(point_idx)

    answer = []

    for idx, frequency in zip(points, frequencies):
        time = first_time + int(idx) * time_step

        answer.append((int(idx), time, float(frequency)))
    
##    with open(output_file, 'w') as f_out:
##        f_out.write("point\tTime\tF0\n")
##        for idx, frequency in zip(points, frequencies):
##            time = first_time + int(idx) * time_step
##            f_out.write(f"{idx}\t{time}\t{frequency}\n")

    return answer

dx = Fraction(1, 50)

def pitchsqueezer_conversion(
    path,
    min_hz=60,
    max_hz=500,
    voicing_thresh=0.2,
    wavelet=False,
    output_format="txt",
    output_directory=None,
    x1=0.0,
    dx=Fraction("1/50"),
    overwrite=False,
    plot=False
):
    """Converts an audio file using pitchsqueezer by calling track_pitch directly."""
    assert path.endswith('.wav')
    np.random.seed(42)
    #pitch_squeezer.np.random.seed(42)
    np.set_printoptions(precision=10, floatmode='fixed')
    
    fullpath = os.path.abspath(path)
    basepath = os.path.splitext(fullpath)[0]
    mrawf0 = basepath + '.mrawf0'
    squeezed_txt = basepath + '.f0.txt'

    # Skip if output already exists and overwrite is False
    if not overwrite and os.path.exists(mrawf0):
        return mrawf0

    # Calculate frame_rate from dx
    frame_rate = float(1 / dx)

    # Run pitch extraction
    try:
        f0, if0 = track_pitch(
            fullpath,
            min_hz=min_hz,
            max_hz=max_hz,
            voicing_thresh=voicing_thresh,
            frame_rate=frame_rate,
            viterbi=True,
            plot=plot
        )
    except Exception as e:
        print(f"Error during pitch extraction: {e}")
        return None

    if wavelet:
        try:
            f0_cwt(if0, plot=plot)
        except Exception as e:
            print(f"Wavelet transform failed: {e}")

    # Save f0 track as text
    try:
        with open(squeezed_txt, 'wt', encoding='utf-8') as txt_file:
            for val in f0:
                print(f"{val:.10f}", file=txt_file)
    except IOError as e:
        print(f"Error writing squeezed f0 file: {e}")
        return None

    # Write mrawf0 output
    try:
        with open(mrawf0, 'wt', encoding='utf-8') as mra:
            print('point\tTime\tF0', file=mra)
            for n, freq in enumerate(f0):
                tim = float(x1 + n * dx)
                if freq > 0:
                    print(n + 1, tim, freq, sep='\t', file=mra)
    except IOError as e:
        print(f"Error writing mrawf0 file: {e}")
        return None

    print(f"Successfully created {mrawf0}")
    return mrawf0

TOLERANCE_LEVEL = 1/(96000*8)

def interpolate_frequency(pitch_data, target_time): #chatgpt
    """
    Interpolates the frequency at a given time based on the pitch data.
    
    Parameters:
    - pitch_data: List of tuples in the format (index+1, time, frequency).
    - target_time: The time at which to interpolate the frequency.
    
    Returns:
    - The interpolated frequency or an error as per the rules.
    """
    # Ensure the list is sorted by time
    ## pitch_data = sorted(pitch_data, key=lambda x: x[1]) # not needed
    
    # Extract times for comparison
    ##times = [t[1] for t in pitch_data] # unnecessary iteration
    
    # Case (c): Requested time is out of range
    if target_time > pitch_data[-1][1]:
        warnings.warn("Error: Requested time is higher than the maximum time in the pitch data. Time difference: {}".format(target_time-pitch_data[-1][1]))
        return 0.0
    
    # Find the closest indices around the target time
    for i, (index, time, frequency) in enumerate(pitch_data):
        if abs(time - target_time) < TOLERANCE_LEVEL: # originally: if time == target_time:
            # Case (a): Time matches exactly
            return frequency
        
        if time > target_time:
            # Found the interval containing target_time
            lower_index = i - 1
            upper_index = i
            
            # Get the surrounding points
            _, t_lower, f_lower = pitch_data[lower_index]
            _, t_upper, f_upper = pitch_data[upper_index]
            
            # Case (b): If either adjacent frequency is 0.0
            if f_lower == 0.0 or f_upper == 0.0:
                return 0.0
            
            # Case (d): Logarithmic interpolation
            log_f_lower = math.log(f_lower)
            log_f_upper = math.log(f_upper)
            
            # Interpolate in the logarithmic domain
            log_f_interpolated = log_f_lower + (log_f_upper - log_f_lower) * (target_time - t_lower) / (t_upper - t_lower)
            return math.exp(log_f_interpolated)
    
    # If the function reaches this point, something unexpected happened
    raise ValueError("Unexpected error: Could not interpolate frequency.")

INFINITY = float('inf')

def semitone_difference(freq1, freq2):
    """Calculate the semitone difference between two frequencies."""
    return abs(12 * math.log2(freq1 / freq2)) if freq1 > 0 and freq2 > 0 else INFINITY

DEFAULT_OCTAVE_THRESHOLD = 9

def detect_creakiness_identifying_points(a1_output, a2_output, semitone_threshold=DEFAULT_OCTAVE_THRESHOLD): # chatgpt fixed this
    """a2_output points where a1_output differs by more than the given threshold and both are positive"""
    creakiness_identifying_points = []

    for i, (_, time, freq_a2) in enumerate(a2_output):
        freq_a1 = interpolate_frequency(a1_output, time)

        # Skip if either A1 or A2 is silent
        if freq_a1 == 0.0 or freq_a2 == 0.0:
            continue

        # Calculate semitone difference
        semitone_diff = semitone_difference(freq_a1, freq_a2)

        # Check if the disagreement qualifies as creaky
        if semitone_diff >= semitone_threshold:
            creakiness_identifying_points.append((time, freq_a2))

    return creakiness_identifying_points

def detect_creaky_regions(a1_output, a2_output, semitone_threshold=DEFAULT_OCTAVE_THRESHOLD, a3_output=None):
    """
    Detect creaky regions based on A1 and A2 outputs, with optional exclusion using A3.

    Parameters:
    - a1_output: List of (index, time, frequency) tuples for A1.
    - a2_output: List of (index, time, frequency) tuples for A2.
    - semitone_threshold: The semitone difference threshold for identifying creaky regions.
    - a3_output: If present, regions that are completely silent according to this algorithm will be excluded.

    Returns:
    - List of (start_time, end_time) tuples for creaky regions in A1 output.
    """
    creaky_regions = []
    creakiness_identifying_points = detect_creakiness_identifying_points(a1_output, a2_output, semitone_threshold)

    # Helper function to expand a creakiness-identifying point
    def expand_region(center_time, a1_output, a2_output, direction="both"):
        start_time, end_time = center_time, center_time

        # Expand forward
        if direction in {"both", "forward"}:
            for i, (_, time, freq_a2) in enumerate(a2_output):
                if time <= center_time:
                    continue  # Start expanding only after the center point
                freq_a1 = interpolate_frequency(a1_output, time)
                semitone_diff = semitone_difference(freq_a1, freq_a2)

                # Stop expansion if A1 is silent or if disagreement is below the threshold
                if freq_a1 == 0.0 or semitone_diff < semitone_threshold:
                    break
                end_time = time

        # Expand backward
        if direction in {"both", "backward"}:
            for i, (_, time, freq_a2) in reversed(list(enumerate(a2_output))):
                if time >= center_time:
                    continue  # Start expanding only before the center point
                freq_a1 = interpolate_frequency(a1_output, time)
                semitone_diff = semitone_difference(freq_a1, freq_a2)

                # Stop expansion if A1 is silent or if disagreement is below the threshold
                if freq_a1 == 0.0 or semitone_diff < semitone_threshold:
                    break
                start_time = time

        return start_time, end_time

    # Helper function to check if a region is silent in A3
    def is_silent_in_a3(start_time, end_time, a3_output):
        sounding_numbers = []
        first_number = None
        last_number = None

        for i, item in enumerate(a3_output):
            number, time, freq = item
            if first_number is None:
                first_number = number
            if time > end_time:
                last_number = number - 1
                break
            if start_time <= time <= end_time and freq > 0.0:
                sounding_numbers.append(number)

        # Handle the case where the region is too short to exclude
        if last_number is not None and last_number - first_number < 3:
            return False

        # Check for consecutive integers in sounding_numbers
        for i in range(len(sounding_numbers) - 1):
            if sounding_numbers[i] + 1 == sounding_numbers[i + 1]:
                return False  # Not silent because a consecutive pair exists

        return True  # Completely silent


    # Build creaky regions
    for i, (center_time, _) in enumerate(creakiness_identifying_points):
        # Expand the region around the creakiness-identifying point
        start_time, end_time = expand_region(center_time, a1_output, a2_output)

        # Handle cases where the point cannot be expanded
        if start_time == center_time and end_time == center_time:
            # Determine the next and previous points
            previous_time = creakiness_identifying_points[i - 1][0] if i > 0 else None
            next_time = creakiness_identifying_points[i + 1][0] if i < len(creakiness_identifying_points) - 1 else None

            # Adjust the region based on neighboring points
            if previous_time is not None:
                start_time = (center_time + previous_time) / 2
            if next_time is not None:
                end_time = (center_time + next_time) / 2

        # Add the expanded region
        creaky_regions.append((start_time, end_time))

    # Merge overlapping or adjacent regions
    merged_regions = []
    for start, end in sorted(creaky_regions):
        if not merged_regions or merged_regions[-1][1] < start:  # No overlap
            merged_regions.append((start, end))
        else:  # Overlapping or adjacent regions
            merged_regions[-1] = (merged_regions[-1][0], max(merged_regions[-1][1], end))

    # Exclude regions that are silent in A3
    if a3_output:
        filtered_regions = [
            (start, end) for start, end in merged_regions if not is_silent_in_a3(start, end, a3_output)
        ]
    else:
        filtered_regions = merged_regions

    return [c for c in filtered_regions if c[1] > c[0]]

def detect_creak(wav_path, step_1=dx, overwrite_1=False, step_2=None, floors=[75.0,75.0,75.0], ceilings=[600.0,600.0,600.0], octave_threshold=DEFAULT_OCTAVE_THRESHOLD,
                 a1='pitchsqueezer', a2='cc', a3='shs', octave_jump_costs=[0.35,0.35,0.35], voicing_thresholds=[0.45,0.45,0.45], voiced_unvoiced_costs=[0.14,0.14,0.14], 
                   very_accurates=[False,False,False], silence_thresholds=[0.03,0.03,0.03], max_candidatess=[15,15,15]):
    methods = [a1, a2, a3]
    if 'pitchsqueezer' in methods: # Not possible to base it on multiple runs of pitchsqueezer
        p_i = methods.index('pitchsqueezer')
        psq = pitchsqueezer_conversion(wav_path, dx=step_1, overwrite=overwrite_1, min_hz=floors[p_i], max_hz=ceilings[p_i], voicing_thresh=voicing_thresholds[p_i])
    else:
        psq = None
        
    if a1 == 'pitchsqueezer':
        a1_output = mrawf0_to_tuple(psq)
    else:
        a1_output = pitch_to_tuple(to_pitch_praat(wav_path, time_step=step_2, pitch_floor=floors[0], pitch_ceiling=ceilings[0], method=a1,
                                                  octave_jump_cost=octave_jump_costs[0], voicing_threshold=voicing_thresholds[0], voiced_unvoiced_cost=voiced_unvoiced_costs[0], 
                   very_accurate=very_accurates[0], silence_threshold=silence_thresholds[0], max_candidates=max_candidatess[0]))
        
    if a2 == 'pitchsqueezer':
        a2_output = mrawf0_to_tuple(psq)
    else:
        a2_output = pitch_to_tuple(to_pitch_praat(wav_path, time_step=step_2, pitch_floor=floors[1], pitch_ceiling=ceilings[1], method=a2,
                                                  octave_jump_cost=octave_jump_costs[1], voicing_threshold=voicing_thresholds[1], voiced_unvoiced_cost=voiced_unvoiced_costs[1], 
                   very_accurate=very_accurates[1], silence_threshold=silence_thresholds[1], max_candidates=max_candidatess[1]))
        
    if a3 == 'pitchsqueezer':
        a3_output = mrawf0_to_tuple(psq)
    elif a3 is None:
        a3_output = None
    else:
        a3_output = pitch_to_tuple(to_pitch_praat(wav_path, time_step=step_2, pitch_floor=floors[2], pitch_ceiling=ceilings[2], method=a3,
                                                  octave_jump_cost=octave_jump_costs[2], voicing_threshold=voicing_thresholds[2], voiced_unvoiced_cost=voiced_unvoiced_costs[2], 
                   very_accurate=very_accurates[1], silence_threshold=silence_thresholds[2], max_candidates=max_candidatess[2]))


    return detect_creaky_regions(a1_output, a2_output, a3_output=a3_output)




def get_wav_duration(wav_file): # ChatGPT
    with wave.open(wav_file, 'rb') as wav:
        num_frames = wav.getnframes()
        frame_rate = wav.getframerate()
        duration = num_frames / float(frame_rate)
    return duration

def process_and_join(lst):
    """
    Processes a list of strings by removing empty edges, converting empty strings in between to commas,
    and returning a space-joined string where commas are appended to the previous word.
    
    Parameters:
    - lst: List of strings
    
    Returns:
    - A formatted string based on the input list
    """
    # Remove empty edges
    if lst and lst[0] == '':
        lst = lst[1:]
    if lst and lst[-1] == '':
        lst = lst[:-1]
    
    # Build the output string
    result = []
    for i, word in enumerate(lst):
        if word == '':
            # Append a comma to the previous word
            if result:
                result[-1] += ','
        else:
            # Add the current word
            result.append(word)
    
    # Return the space-joined result
    return ' '.join(result)


def find_wav_text(wav):
    try:
        txt = read_tg(wav[:-4] + '.TextGrid')[0]
        return '"' + process_and_join([w[-1] for w in txt]) + '"'
    except FileNotFoundError:
        with open(wav[:-4] + '.txt', 'rt', encoding='utf-8') as k:
            c = k.read()
            return '"' + ' '.join(c[c.startswith('\ufeff'):].split()) + '"'

def add_missing_intervals(intervals, nonmissing='n', missing=''): # chatgpt
    """
    Adds missing intervals and labels them with `missing`, while labeling existing intervals with `nonmissing`.
    
    Parameters:
    - intervals: List of tuples representing existing intervals (start_time, end_time).
    - nonmissing: Label for non-missing intervals.
    - missing: Label for missing intervals.
    
    Returns:
    - List of tuples [(start_time, end_time, label)] with missing and non-missing intervals labeled.
    """
    if not intervals:
        return []
    
    # Output list
    result = []
    
    # Starting from time 0
    previous_end = 0
    
    for start, end in intervals:
        # Add missing interval if there's a gap
        if start > previous_end:
            result.append((previous_end, start, missing))
        
        # Add non-missing interval
        result.append((start, end, nonmissing))
        
        # Update the end of the last interval
        previous_end = end
    
    return result


def generate_tg_string(wav, creaky_segments):

    duration = get_wav_duration(wav)

    try:
        pass
    except:
        pass

    try:
        text = find_wav_text(wav)
    except FileNotFoundError:
        text = '"<teksti>"'

    tg_start = r'''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0 
xmax = {0} 
tiers? <exists> 
size = 1 
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "creaks" 
        xmin = 0 
        xmax = {0}
'''.format(duration, text)
    interval_values = add_missing_intervals(creaky_segments)
    #print(interval_values)
    if interval_values:
        interval_values.append((interval_values[-1][1], duration, ''))
    else:
        interval_values.append((0, duration, ''))
    #print(interval_values)
    interval_strings = ['        intervals: size = {} \n'.format(len(interval_values))]
    creak_tier_interval_template = '''        intervals [{}]:
            xmin = {} 
            xmax = {} 
            text = "{}"
'''
    for intervalm, t in enumerate(interval_values):
        if t[1] > duration:
            interval_strings.append(creak_tier_interval_template.format(intervalm+1, t[0], duration, t[2]))
            interval_strings[0] = '        intervals: size = {} \n'.format(intervalm+1)
            break
        else:
            interval_strings.append(creak_tier_interval_template.format(intervalm+1, t[0], t[1], t[2]))
    
    return tg_start + ''.join(interval_strings)


def write_textgrid(wav, creaky_segments, path):
    with open(path, 'w', encoding='utf-16-be') as file:
        file.write('\ufeff')
        file.write(generate_tg_string(wav, creaky_segments))


def write_creak_labeled_mrawf0(mrawf0, creaky_segments, output_file):
    answer = []

    with open(mrawf0, 'rt', encoding='utf-8') as k:
        b = k.read()

    for item in b.splitlines():
        try:
            n, t, f = item.split('\t')
            tf = float(t)
            if any((segment[0] <= tf <= segment[1]) for segment in creaky_segments):
                ff = 1
            else:
                ff = float(f)
            answer.append('{}\t{}\t{}'.format(n, t, ff))
        except ValueError:
            answer.append(item)

    with open(output_file, 'wt', encoding='utf-8') as g:
        print('\n'.join(answer), end='', file=g)
    


def write_and_label(wav_path, floors=75.0, ceilings=600.0, octave_threshold=DEFAULT_OCTAVE_THRESHOLD, a1='pitchsqueezer', a2='cc', a3='shs',
                    octave_jump_costs=0.35, voicing_thresholds=0.45, voiced_unvoiced_costs=0.14, 
                   very_accurates=False, silence_thresholds=0.03, max_candidatess=15):
    if isinstance(floors, (float, int, Fraction)):
        floors = [floors, floors, floors]
    if isinstance(ceilings, (float, int, Fraction)):
        ceilings = [ceilings, ceilings, ceilings]
    if isinstance(octave_jump_costs, (float, int, Fraction)):
        octave_jump_costs = [octave_jump_costs, octave_jump_costs, octave_jump_costs]
    if isinstance(voicing_thresholds, (float, int, Fraction)):
        voicing_thresholds = [voicing_thresholds, voicing_thresholds, voicing_thresholds]
    if isinstance(voiced_unvoiced_costs, (float, int, Fraction)):
        voiced_unvoiced_costs = [voiced_unvoiced_costs, voiced_unvoiced_costs, voiced_unvoiced_costs]
    if isinstance(very_accurates, (float, int, Fraction)):
        very_accurates = [very_accurates, very_accurates, very_accurates]
    if isinstance(silence_thresholds, (float, int, Fraction)):
        silence_thresholds = [silence_thresholds, silence_thresholds, silence_thresholds]
    if isinstance(max_candidatess, (float, int, Fraction)):
        max_candidatess = [max_candidatess, max_candidatess, max_candidatess]

    if a1 == 'pitchsqueezer':
        floors[0] = math.floor(floors[0])
        ceilings[0] = math.ceil(ceilings[0])
    if a2 == 'pitchsqueezer':
        floors[1] = math.floor(floors[1])
        ceilings[1] = math.ceil(ceilings[1])
    if a3 == 'pitchsqueezer':
        floors[2] = math.floor(floors[2])
        ceilings[2] = math.ceil(ceilings[2])
    
    creaky_regions = detect_creak(wav_path, floors=floors, ceilings=ceilings, octave_threshold=octave_threshold, a1=a1, a2=a2, a3=a3,
                                  octave_jump_costs=octave_jump_costs, voicing_thresholds=voicing_thresholds, voiced_unvoiced_costs=voiced_unvoiced_costs, 
                   very_accurates=very_accurates, silence_thresholds=silence_thresholds, max_candidatess=max_candidatess)
    write_textgrid(wav_path, creaky_regions, wav_path[:-4] + '_creaks.TextGrid')

    if 'pitchsqueezer' in {a1, a2, a3}:
        write_creak_labeled_mrawf0(wav_path[:-4] + '.mrawf0', creaky_regions, wav_path[:-4] + '_withcreaks.mrawf0')

if __name__ == '__main__':

    os.chdir(output_folder)

# On average, these parameters worked best as presented by Haakana & Jokinen (2025)
floors = [50, 60, 60]
ceilings = [450, 450, 450]
a1 = 'pitchsqueezer'
a2 = 'cc'
a3 = None
octave_jump_costs = [0.35, 0.35, 0.35]
voicing_thresholds = [0.2, 0.45, 0.45]
voiced_unvoiced_costs = [0.14, 0.14, 0.14]
very_accurates = [False, False, False]
silence_thresholds = [0.03, 0.03, 0.03]
max_candidatess = [15, 15, 15]

if __name__ == '__main__':

    for file in [w for w in os.listdir() if w.lower().endswith('.wav')]:
        #raise
        write_and_label(file, floors, ceilings, a1=a1, a2=a2, a3=a3, octave_jump_costs=octave_jump_costs, voicing_thresholds=voicing_thresholds, voiced_unvoiced_costs=voiced_unvoiced_costs, 
                       very_accurates=very_accurates, silence_thresholds=silence_thresholds, max_candidatess=max_candidatess)
