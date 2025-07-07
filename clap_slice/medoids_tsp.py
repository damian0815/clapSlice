import fast_tsp, kmedoids
import numpy as np
import torch


def solve_tsp_fasttsp(distance_matrix):
    min_value = distance_matrix.min()
    max_value = distance_matrix.max()
    range = max_value - min_value
    scale_factor = 65535/range
    distance_matrix_int = ((distance_matrix-min_value) * scale_factor).cpu().numpy().astype(np.int64)
    return fast_tsp.find_tour(distance_matrix_int)


def k_medoids(embeddings, k):
    distance_matrix = get_distance_matrix(embeddings)
    result = kmedoids.fasterpam(distance_matrix.to('cpu', dtype=torch.float32).numpy(), k)
    # print(result)
    return (
        torch.from_numpy(result.medoids.astype(np.int64)),
        torch.from_numpy(result.labels.astype(np.int64))
    )


def get_distance_matrix(embeddings_a, embeddings_b=None):
    if embeddings_b is None:
        embeddings_b = embeddings_a

    similarity_matrix = torch.mm(embeddings_a, embeddings_b.t())
    #print(similarity_matrix.min(), similarity_matrix.max())
    similarity_matrix = similarity_matrix * 0.5 + 0.5
    distance_matrix = 1 - similarity_matrix
    #print(distance_matrix.min(), distance_matrix.max())
    #distance_matrix.fill_diagonal_(0)

    return distance_matrix


def sort_tsp(embeddings,
             indices:torch.IntTensor|None=None,
             cluster_assignment=None,
             dist_matrix_offset=None,
             pin_first_index: int=None,
             pin_last_index: int=None,
             ) -> torch.IntTensor|tuple[torch.IntTensor, torch.Tensor]:
    if indices is None:
        indices = torch.arange(embeddings.shape[0])
    medoids_distance_matrix = get_distance_matrix(embeddings[indices])
    if dist_matrix_offset is not None:
        medoids_distance_matrix += dist_matrix_offset

    large_distance = medoids_distance_matrix.max() * 10
    #if preserve_ends:
    #    medoids_distance_matrix[0, -1] = large_distance
    #    medoids_distance_matrix[-1, 0] = large_distance
    #print("pin:", pin_first_index, pin_last_index, medoids_distance_matrix[:, pin_first_index])
    if pin_first_index is not None:
        medoids_distance_matrix[:, pin_first_index] += large_distance
    if pin_last_index is not None:
        medoids_distance_matrix[pin_last_index, :] += large_distance
    #print("after pin:", pin_first_index, pin_last_index, medoids_distance_matrix[:, pin_first_index])
    medoids_distance_matrix.fill_diagonal_(0)

    # route = solve_mtsp_dynamic_programming(medoids_distance_matrix, 1)
    # route = solve_tsp_simulated_annealing(medoids_distance_matrix)
    print("computing route")
    #print(medoids_distance_matrix)
    route = solve_tsp_fasttsp(medoids_distance_matrix)

    indices_sorted = indices[route]
    if cluster_assignment is None:
        return indices_sorted
    else:
        return indices_sorted, apply_route_to_cluster_assignment(route, cluster_assignment)


def apply_route_to_cluster_assignment(route, cluster_assignment) -> torch.Tensor:
    inv_route = {}
    for pos, idx in enumerate(route):
        inv_route[idx] = pos

    cluster_assignment_sorted = [
        inv_route[assignment]
        for assignment in cluster_assignment.tolist()
    ]
    cluster_assignment_sorted = torch.tensor(cluster_assignment_sorted)

    return cluster_assignment_sorted
