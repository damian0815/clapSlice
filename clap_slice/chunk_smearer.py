import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Literal

import torch


@dataclass
class SmearDetails:
    source_chunk_index: int
    source_amplitude: float
    spread_slot_pct: float
    priority: float
    ramp_type: Literal['ramp_in', 'ramp_in_out', 'ramp_out', 'none']

    def __repr__(self):
        return f"Smear({self.source_chunk_index}({self.spread_slot_pct} * {self.source_amplitude}{', ' + self.ramp_type if self.ramp_type != 'none' else ''}))"


def get_smear_source_list(
        num_source_chunks: int,
        sort_order: list[int]|torch.IntTensor,
        smear_width: int = 0,
        spread: int = 0,
        wrap_mode: Literal['wrap', 'bleed', 'cut'] = 'wrap',
        consolidate_smears = True,
        smear_mode: Literal['in', 'in-out'] = 'in-out',
        dynamic_width_cb: Callable[[int], tuple[int, int]]|None = None,
        envelope_shape: Literal['cos_2pi', 'sin_pi', 'log'] = 'cos_2pi'
) -> list[list[SmearDetails]]:

    """
    "Smear" a set of source chunks over an output range in a given order.

    :param num_source_chunks: Number of chunks in the source
    :param sort_order: Order the chunks should appear in the output (need not be complete coverage)
    :param smear_width: Controls repetition. If 0, each source chunk is used only once. If N, a source chunk is repeated N times on either side of its target position (with cos fade envelope)
    :param spread: Controls source width. If >0, audio from either side of the source chunk will also be output for each chunk.
    :param wrap_mode: What to do about smears that spill over the original audio bounds
    :param consolidate_smears: If True, simplify output by merging duplicate source references.
    :param smear_mode: Whether to smear on both sides of the target index (deafult) or just one side.
    :param dynamic_width_cb: If specified, called once per source chunk to specify smear width and spread dynamically. The callback should return a pair of ints (smear_width, spread).
    :param envelope_shape: Shape of envelope
    :return:
    """


    if isinstance(sort_order, torch.Tensor):
        sort_order = sort_order.tolist()
    print(sort_order)
    num_target_chunks = len(sort_order)

    envelope = _build_envelope(envelope_shape, smear_width, smear_mode)
    #print(f"smear width {smear_width}, envelope {envelope_shape} -> {envelope}")
    smear_source_dict: dict[int, list[SmearDetails]] = defaultdict(list)

    for position_idx, source_chunk_idx in enumerate(sort_order):
        if dynamic_width_cb is not None:
            smear_width, spread = dynamic_width_cb(source_chunk_idx)
            smear_width = round(smear_width)
            spread = round(spread)
            envelope = _build_envelope(envelope_shape, smear_width, smear_mode)

        for smear_slot in range(-smear_width, smear_width + 1):
            for spread_slot in range(-spread, spread + 1):
                priority = -abs(smear_slot) + -abs(spread_slot/2)
                this_source_chunk_idx = source_chunk_idx + spread_slot
                if wrap_mode == 'wrap':
                    this_source_chunk_idx = this_source_chunk_idx % num_source_chunks
                elif this_source_chunk_idx < 0 or this_source_chunk_idx >= num_source_chunks:
                        continue

                if spread == 0:
                    ramp_type = 'ramp_in_out'
                elif spread_slot == -spread:
                    ramp_type = 'ramp_in'
                elif spread_slot == spread:
                    ramp_type = 'ramp_out'
                else:
                    ramp_type = 'none'

                smear_details = SmearDetails(
                    source_chunk_index=this_source_chunk_idx,
                    source_amplitude=envelope[smear_slot + smear_width],
                    spread_slot_pct=(spread_slot / spread if spread > 0 else 0),
                    priority=priority,
                    ramp_type=ramp_type
                )
                target_index = position_idx + smear_slot + spread_slot
                if wrap_mode == 'wrap':
                    target_index = target_index % num_target_chunks
                elif wrap_mode == 'cut':
                    if target_index < 0 or target_index >= num_target_chunks:
                        continue

                #print(f"adding {smear_details.source_chunk_index} * {smear_details.amplitude} to {target_index} (source from {(source_chunk_idx, spread_slot)}, target from {(position_idx, smear_slot, spread_slot)}, num_chunks {num_chunks})")
                smear_source_dict[target_index].append(smear_details)
        #break
    smear_source_list = [v for k, v in sorted(smear_source_dict.items(), key=lambda kv: kv[0])]
    if not consolidate_smears:
        return smear_source_list
    else:
        return [_consolidate_smears(smears) for smears in smear_source_list]


def _build_envelope(envelope_shape, smear_width: float, smear_mode) -> torch.Tensor:
    steps = 2*smear_width+1 + 2
    if envelope_shape == 'cos_2pi':
        envelope = 0.5*(1-torch.cos(torch.linspace(0, 2*math.pi, steps=steps)))[1:-1]
    elif envelope_shape == 'sin_pi':
        envelope = torch.sin(torch.linspace(0, math.pi, steps=steps))[1:-1]
    elif envelope_shape == 'log':
        log_in = (torch.logspace(0, 1, steps=smear_width+2, base=10)/10).tolist()[1:-1]
        envelope = torch.tensor(log_in + [1] + list(reversed(log_in)))
    else:
        raise ValueError(f'Unknown envelope shape: {envelope_shape}')
    if smear_mode == 'in':
        for i in range(smear_width+1, 2*smear_width+1):
            envelope[i] = 0
    return envelope


def _consolidate_smears(smears: list[SmearDetails]) -> list[SmearDetails]:
    unique_sources = sorted({sd.source_chunk_index for sd in smears})
    consolidated_smears = []
    for source_index in unique_sources:
        smears_to_consolidate = [sd for sd in smears
                                 if sd.source_chunk_index == source_index]
        envelope_amplitude = sum(sd.source_amplitude for sd in smears_to_consolidate)
        priority = max(sd.priority for sd in smears_to_consolidate)
        ramp_type = max(smears_to_consolidate, key=lambda sd: sd.soeurce_amplitude).ramp_type
        spread_slot_pct = min(sd.spread_slot_pct for sd in smears_to_consolidate)
        consolidated_smears.append(SmearDetails(
            source_chunk_index=source_index,
            source_amplitude=envelope_amplitude,
            spread_slot_pct=spread_slot_pct,
            priority=priority,
            ramp_type=ramp_type
        ))
    return consolidated_smears



