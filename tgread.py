def bigrams(iterable): # ChatGPT's pure python implementation of nltk.bigrams (one less dependency)
    it = iter(iterable)
    try:
        prev = next(it)
    except StopIteration:
        return  # empty iterable, do nothing
    for item in it:
        yield (prev, item)
        prev = item

def _default_to_self(key, d):
    if key in d:
        return d[key]
    return key

def number(x):
    assert type(x) == str
    try:
        return int(x)
    except ValueError:
        return float(x)

def _original_open_tg(tg):
    try:
        with open(tg, 'rt', encoding='utf-16') as k:
            return k.read()
    except UnicodeError:
        with open(tg, 'rt', encoding='utf-8') as k:
            b = k.read()
            return b[b.startswith('\ufeff'):]

def open_tg(tg): # chatgpt's improvement, also removes '\x1f'
    encodings = ['utf-16', 'utf-8-sig', 'utf-8']  # utf-8-sig handles BOM automatically
    for enc in encodings:
        try:
            with open(tg, 'rt', encoding=enc) as f:
                content = f.read()
                return content.replace('\x1f', '')  # Remove any occurrences of '\x1f'
        except UnicodeError:
            continue  # Try the next encoding
    raise UnicodeError(f"Could not decode the TextGrid file {tg} with available encodings.")

def find_intervals_of_single_tier(tg_lines_section, replacements={}):
    answer = []

    for n, i in enumerate(tg_lines_section):
        if i.startswith('            text = '):
            assert n >= 2
            text = i[20:].rstrip()[:-1]
            xmax_text = tg_lines_section[n-1][19:].rstrip()
            xmax = number(xmax_text)
            xmin_text = tg_lines_section[n-2][19:].rstrip()
            xmin = number(xmin_text)
            answer.append((xmin, xmax, _default_to_self(text, replacements)))

    return answer

def find_intervals(tg_text, replacements={}):
    answer = []
    
    tg_lines = tg_text.splitlines()

    first_interval_starters = [n for n, i in enumerate(tg_lines) if i == '        intervals [1]:']

    for start, stop in bigrams(first_interval_starters):
        answer.append(find_intervals_of_single_tier(tg_lines[start:stop], replacements))

    answer.append(find_intervals_of_single_tier(tg_lines[first_interval_starters[-1]:], replacements))

    return answer

def read_tg(filename, replacements={}):
    return find_intervals(open_tg(filename), replacements)

def generate_tg_text(tiers, tier_names):
    end_time = tiers[0][-1][1]
    start_time = tiers[0][0][0]

    startt = r'''File type = "ooTextFile"
Object class = "TextGrid"

xmin = {} 
xmax = {} 
tiers? <exists> 
size = {} 
item []: '''.format(start_time, end_time, len(tiers))

    tier_texts = []

    for n, tier in enumerate(tiers):
        start_of_this = '''    item [{}]:
        class = "IntervalTier" 
        name = "{}" 
        xmin = {} 
        xmax = {} 
        intervals: size = {} '''.format(n+1, tier_names[n], start_time, end_time, len(tier))

        interval_texts = []

        for m, interval in enumerate(tier):
            start, end, text = interval
            interval_text = r'''        intervals [{}]:
            xmin = {} 
            xmax = {} 
            text = "{}" '''.format(m+1, start, end, text)

            interval_texts.append(interval_text)
            
        tier_texts.append(start_of_this + '\n' + '\n'.join(interval_texts))

    return startt + '\n' + '\n'.join(tier_texts) + '\n'

def write_tg(filename, tiers, tier_names, encoding='utf-8'):
    text = generate_tg_text(tiers, tier_names)
    with open(filename, 'wt', encoding=encoding) as k:
        print(text, end='', file=k)

def fill_labels(intervals, fill='', end=None): # chatgpt
    """
    Fills missing intervals with a given string and optionally appends an interval up to a specified end time.

    Parameters:
        intervals (list of tuples): List of (start_time, end_time, label).
        fill (str): The label to use for filling missing intervals (default is empty string).
        end (float, optional): If provided, ensures the last interval extends to this time.

    Returns:
        list of tuples: A new list with gaps filled and an optional final interval.
    """
    if not intervals:
        return [(0, end, fill)] if end is not None else []

    filled_intervals = []
    prev_end = 0  # Start from zero

    for start, stop, label in sorted(intervals):
        # Fill the gap before the current interval
        if start > prev_end:
            filled_intervals.append((prev_end, start, fill))
        # Add the actual interval
        filled_intervals.append((start, stop, label))
        prev_end = stop  # Update the last seen end time

    # If 'end' is specified and the last interval doesn't reach it, add a final filler
    if end is not None and prev_end < end:
        filled_intervals.append((prev_end, end, fill))

    return filled_intervals

def generate_pointtier_text(points, length, name): # textgrid with only a single point tier
    start = r'''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0 
xmax = {0} 
tiers? <exists> 
size = 1 
item []: 
    item [1]:
        class = "TextTier" 
        name = "{2}" 
        xmin = 0 
        xmax = {0} 
        points: size = {1} '''.format(length, len(points), name)

    point_strs = []

    for n, point in enumerate(points):
        point_str = r'''        points [1]:
            number = {} 
            mark = "{}" '''.format(point[0], point[1])
        point_strs.append(point_str)

    return start + '\n' + '\n'.join(point_strs) + '\n'

def write_pointtier_tg(points, length, name, output_file, encoding='utf-8'):
    """Writes textgrid with single point tier. Parameters:
- points: list of tuples of (timestamp, label) format
- length: length of audio file (xmax)
- name: name of tier
- output_file: name of output file
- encoding (optional, default 'utf-8'): encoding in which the TextGrid is written"""
    text = generate_pointtier_text(points, length, name)
    with open(output_file, 'wt', encoding=encoding) as k:
        print(text, end='', file=k)
