import argparse
import csv

import json
from itertools import product


# Chromatic → Circle of Fifths position
def to_cof(semitone):
    return (semitone * 7) % 12


# CoF distance between two semitone keys
def cof_distance(a, b, mode_a='major', mode_b='major'):
    # Relative minor is 3 semitones below its relative major
    # Treat relative pairs as distance 0
    a_major = a if mode_a == 'major' else (a + 3) % 12
    b_major = b if mode_b == 'major' else (b + 3) % 12
    d = abs(to_cof(a_major) - to_cof(b_major))
    return min(d, 12 - d)


# Find optimal shifts for a collection of songs
def find_optimal_shifts(songs, max_shift=6, alpha=1.0, beta=2.0):
    """
    songs: list of dicts with 'name', 'key' (0-11, C=0), 'mode' ('major'/'minor')
    max_shift: max semitones to shift (±)
    alpha: penalty weight for shift magnitude
    beta: penalty weight for harmonic distance between songs
    Returns: list of (song_name, shift, resulting_key)
    """
    n = len(songs)
    shift_range = range(-max_shift, max_shift + 1)

    best_cost = float('inf')
    best_shifts = [0] * n

    # For small n (≤ ~8), brute force is tractable
    # For larger n, use the centroid heuristic below
    if n <= 8:
        for shifts in product(shift_range, repeat=n):
            shifted_keys = [((songs[i]['key'] + shifts[i]) % 12, songs[i]['mode'])
                            for i in range(n)]

            shift_cost = alpha * sum(abs(s) for s in shifts)
            harmony_cost = beta * sum(
                cof_distance(shifted_keys[i][0], shifted_keys[j][0],
                             shifted_keys[i][1], shifted_keys[j][1])
                for i in range(n) for j in range(i + 1, n)
            )
            cost = shift_cost + harmony_cost
            if cost < best_cost:
                best_cost = cost
                best_shifts = list(shifts)
    else:
        # Centroid heuristic for larger collections
        best_cost, best_shifts = centroid_heuristic(songs, max_shift, alpha, beta)

    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    results = []
    for i, s in enumerate(best_shifts):
        new_key = (songs[i]['key'] + s) % 12
        results.append({
            'name': songs[i]['name'],
            'original_key': f"{key_names[songs[i]['key']]} {songs[i]['mode']}",
            'shift': s,
            'new_key': f"{key_names[new_key]} {songs[i]['mode']}",
        })
    return results, best_cost


def centroid_heuristic(songs, max_shift, alpha, beta):
    """For larger collections: find the CoF 'center of gravity' and shift minimally."""
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    n = len(songs)

    best_cost = float('inf')
    best_shifts = [0] * n

    # Try each possible target "tonic" (0-11) as the center
    for target in range(12):
        shifts = []
        for song in songs:
            key = song['key']
            if song['mode'] == 'minor':
                song['key'] = (key + 3) % 12  # treat as relative major
            # Find minimum shift to bring CoF position close to target
            options = [(s, cof_distance((song['key'] + s) % 12, target))
                       for s in range(-max_shift, max_shift + 1)]
            best_s = min(options, key=lambda x: alpha * abs(x[0]) + beta * x[1])[0]
            shifts.append(best_s)

        shifted = [((songs[i]['key'] + shifts[i]) % 12, songs[i]['mode']) for i in range(n)]
        cost = (alpha * sum(abs(s) for s in shifts) +
                beta * sum(cof_distance(shifted[i][0], shifted[j][0],
                                        shifted[i][1], shifted[j][1])
                           for i in range(n) for j in range(i + 1, n)))
        if cost < best_cost:
            best_cost = cost
            best_shifts = shifts[:]

    return best_cost, best_shifts


def _parse_key_sig(key_sig: str) -> tuple[int, str]:
    keys_to_semis_map = {
        'C': 0,
        'C#': 1,
        'Db': 1,
        'D': 2,
        'D#': 3,
        'Eb': 3,
        'E': 4,
        'F': 5,
        'F#': 6,
        'Gb': 6,
        'G': 7,
        'G#': 8,
        'Ab': 8,
        'A': 9,
        'A#': 10,
        'Bb': 10,
        'B': 11
    }
    parts = key_sig.split(' ')
    return keys_to_semis_map[parts[0]], parts[1].lower()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv_path", type=str)
    parser.add_argument("--output_json_path", type=str, default=None)

    args = parser.parse_args()

    songs = []

    with open(args.input_csv_path) as f:
        reader = csv.reader(f)
        for line in reader:
            key_sig = line[0] # eg "A# Major"
            path = line[1]

            key, mode = _parse_key_sig(key_sig)
            songs.append(dict(
                name=path,
                key=key,
                mode=mode,
                key_sig_str=key_sig
            ))

    results, cost = find_optimal_shifts(songs, max_shift=3, alpha=1.0, beta=2.0)
    
    if args.output_json_path is not None:
        with open(args.output_json_path, 'w') as f:
            json.dump(results, f)
    else:
        for r in results:
            print(f"{r['name']}: {r['original_key']} → shift {r['shift']:+d} → {r['new_key']}")
        print(f"Total cost: {cost:.2f}")
