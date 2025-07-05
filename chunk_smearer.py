import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch


@dataclass
class SmearDetails:
    source_chunk_index: int
    amplitude: float
    priority: float

    def __repr__(self):
        return f"Smear({self.source_chunk_index} * {self.amplitude})"


def get_smear_source_list(
        num_chunks: int,
        sort_order: list[int],
        smear_width: int,
        spread = 4,
        smear_mode: Literal['in', 'in-out'] = 'in-out',
        dynamic_smear_width_cb: Callable[[int], int]|None = None
) -> list[list[SmearDetails]]:

    if isinstance(sort_order, torch.Tensor):
        sort_order = sort_order.tolist()
    print(sort_order)

    def get_envelope(smear_width):
        envelope = 0.5*(1-torch.cos(torch.linspace(0, 2*math.pi, 2*smear_width+1)))
        if smear_mode == 'in':
            for i in range(smear_width+1, 2*smear_width+1):
                envelope[i] = 0
        return envelope

    envelope = get_envelope(smear_width)
    smear_source_list: list[list[SmearDetails]] = [[] for _ in range(num_chunks)]

    for position_idx in range(num_chunks):
        source_chunk_idx = sort_order[position_idx]

        if dynamic_smear_width_cb is not None:
            smear_width = dynamic_smear_width_cb(source_chunk_idx)
            envelope = get_envelope(smear_width)

        common_chunk_amplitude_factor = (1 / ((spread + 1) * smear_width))

        for smear_slot in range(-smear_width, smear_width + 1):
            for spread_slot in range(-spread, spread + 1):
                priority = -abs(smear_slot) + -abs(spread_slot/2)
                smear_details = SmearDetails(
                    source_chunk_index=_wrap(source_chunk_idx + spread_slot, num_chunks),
                    amplitude=envelope[smear_slot + smear_width] * common_chunk_amplitude_factor,
                    priority=priority
                )
                target_index = _wrap(position_idx + smear_slot + spread_slot, num_chunks)
                #print(f"adding {smear_details.source_chunk_index} * {smear_details.amplitude} to {target_index} (source from {(source_chunk_idx, spread_slot)}, target from {(position_idx, smear_slot, spread_slot)}, num_chunks {num_chunks})")
                smear_source_list[target_index].append(smear_details)
        #break
    return [_consolidate_smears(smears) for smears in smear_source_list]


def _consolidate_smears(smears: list[SmearDetails]) -> list[SmearDetails]:
    source_indices = sorted({sd.source_chunk_index for sd in smears})
    consolidated_smears = []
    for source_index in source_indices:
        amplitude = sum(sd.amplitude for sd in smears
                        if sd.source_chunk_index == source_index)
        priority = max(sd.priority for sd in smears
                       if sd.source_chunk_index == source_index)
        consolidated_smears.append(SmearDetails(
            source_chunk_index=source_index, amplitude=amplitude, priority=priority
        ))
    return consolidated_smears




def _wrap(idx, total):
    if idx < 0:
        return total + idx
    elif idx >= total:
        return idx - total
    else:
        return idx