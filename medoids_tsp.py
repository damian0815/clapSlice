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


def sort_tsp(embeddings, indices=None, cluster_assignment=None, dist_matrix_offset=None):
    if indices is None:
        indices = torch.arange(embeddings.shape[0])
    print("computing distance matrix")
    medoids_distance_matrix = get_distance_matrix(embeddings[indices])
    if dist_matrix_offset is not None:
        medoids_distance_matrix += dist_matrix_offset
    # route = solve_mtsp_dynamic_programming(medoids_distance_matrix, 1)
    # route = solve_tsp_simulated_annealing(medoids_distance_matrix)
    print("computing route")
    #print(medoids_distance_matrix)
    route = solve_tsp_fasttsp(medoids_distance_matrix)
    # solve_tsp_nearest_neighbor(medoids_distance_matrix)
    indices_sorted = indices[route]
    if cluster_assignment is None:
        return indices_sorted

    inv_route = torch.unique(torch.tensor(route), return_inverse=True)[1]
    inv_route = {}
    for pos, idx in enumerate(route):
        inv_route[idx] = pos

    cluster_assignment_sorted = [
        inv_route[assignment]
        for assignment in cluster_assignment.tolist()
    ]
    cluster_assignment_sorted = torch.tensor(cluster_assignment_sorted)

    return indices_sorted, cluster_assignment_sorted
