import numpy as np
from scipy.linalg import expm, expm_frechet
import LineGraph as lg
import Tools as tools

# region: Updating and downdating techniques for optimizing network communicability / Edge Modification Criteria for Enhancing the Communicability of Digraphs
def edge_total_communicability_centrality(adj_matrix, directed=False):
    '''Calculates the edge total communicability centrality of all edges and virtual edges in a network.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
            directed: boolean
                specifies if network is directed or undirected
        Returns:
            edge_total_comm_centr_downdating: ndarray
                the values of edge total communicability centrality for all edges of the network
            edge_total_comm_centr_updating: ndarray
                the values of edge total communicability centrality for all virtual edges of the network
            edge_list_downdating: list
                the edge list of all edges of the network
            edge_list_updating: list
                the edge list of all virtual edges of the network'''
    if directed:
        U, S, V = np.linalg.svd(adj_matrix)
        # maybe unnecessary because if the singular values with index > r are equal to 0, than all of the columns of U and V with index > r will have no contribution to the total_hub/authority_comm because of multiplication with 0 ---------------
        r = np.linalg.matrix_rank(adj_matrix) # also computable via: r = np.argmin(S) (if performance is an issue)
        U = U[:, :r]
        V = V[:r, :]
        S = S[:r]
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Sigma = np.diag(S)

        total_hub_communicability = U @ np.sinh(Sigma) @ V
        total_hub_communicability = tools.sum_rows_and_reshape(total_hub_communicability)

        total_authority_communicability = V.T @ np.sinh(Sigma) @ U.T
        total_authority_communicability = tools.sum_rows_and_reshape(total_authority_communicability)

        edge_total_comm_centr = np.dot(total_hub_communicability, total_authority_communicability.T)

        tmp_adj_matrix_downdating = np.where(adj_matrix != 0, 1, np.nan)
        tmp_adj_matrix_updating = np.where(((adj_matrix == 0) & ~np.eye(adj_matrix.shape[0], dtype=bool)), 1, np.nan)

        edge_total_comm_centr_downdating = np.multiply(edge_total_comm_centr, tmp_adj_matrix_downdating)
        edge_total_comm_centr_downdating = edge_total_comm_centr_downdating.flatten()
        edge_total_comm_centr_updating = np.multiply(edge_total_comm_centr, tmp_adj_matrix_updating)
        edge_total_comm_centr_updating = edge_total_comm_centr_updating.flatten()

    elif not directed:
        exp_adj = expm(adj_matrix)        
        row_sum = tools.sum_rows_and_reshape(exp_adj)
        edge_total_comm_centr = np.dot(row_sum, row_sum.T)

        tmp_adj_matrix_downdating = np.where(adj_matrix != 0, 1, np.nan)
        tmp_adj_matrix_updating = np.where(((adj_matrix == 0) & ~np.eye(adj_matrix.shape[0], dtype=bool)), 1, np.nan)

        edge_total_comm_centr_downdating = np.multiply(edge_total_comm_centr, tmp_adj_matrix_downdating)
        edge_total_comm_centr_downdating = edge_total_comm_centr_downdating[np.triu_indices(adj_matrix.shape[0])]
        edge_total_comm_centr_updating = np.multiply(edge_total_comm_centr, tmp_adj_matrix_updating)
        edge_total_comm_centr_updating = edge_total_comm_centr_updating[np.triu_indices(adj_matrix.shape[0])]

    edge_total_comm_centr_downdating = edge_total_comm_centr_downdating[~np.isnan(edge_total_comm_centr_downdating)]
    edge_total_comm_centr_updating = edge_total_comm_centr_updating[~np.isnan(edge_total_comm_centr_updating)]

    adj_matrix_edge_list_downdating = adj_matrix.copy()
    adj_matrix_edge_list_updating = np.where(((adj_matrix == 0) & ~np.eye(adj_matrix.shape[0], dtype=bool)), 1, 0)
    edge_list_downdating = tools.generate_edge_list(adj_matrix_edge_list_downdating, directed)
    edge_list_updating = tools.generate_edge_list(adj_matrix_edge_list_updating, directed)

    return edge_total_comm_centr_downdating, edge_total_comm_centr_updating, edge_list_downdating, edge_list_updating
# endregion

# region: Edge importance in a network via line graph
def edge_line_graph_centrality(adj_matrix, directed=False):
    '''Calculates the edge line graph centrality of all edges in a network.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
            directed: boolean
                specifies if network is directed or undirected
        Returns:
            edge_line_graph_centr: ndarray
                the values of edge total communicability centrality for all edges of the network
            edge_list: list
                the edge list of all edges of the network'''
    if directed:
        adj_matrix_line_graph = lg.line_graph_adj_matrix_directed(adj_matrix)
        exp_adj = expm(adj_matrix_line_graph)
        edge_line_graph_centr_transmitter = np.sum(exp_adj, 1)
        edge_line_graph_centr_receiver = np.sum(exp_adj, 0)
        edge_line_graph_centr = edge_line_graph_centr_transmitter * edge_line_graph_centr_receiver
    else:
        adj_matrix_line_graph = lg.line_graph_adj_matrix(adj_matrix)
        exp_adj = expm(adj_matrix_line_graph)
        edge_line_graph_centr = np.sum(exp_adj, axis=1)

    edge_list = tools.generate_edge_list(adj_matrix, directed)

    return edge_line_graph_centr, edge_list

def total_line_graph_centrality(adj_matrix, directed=False):
    '''Calculates the total line graph centrality of a network.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
            directed: boolean
                specifies if network is directed or undirected
        Returns:
            edge_total_line_graph_centr: float
                the edge total line graph centrality of the network'''
    edge_line_graph_centr = edge_line_graph_centrality(adj_matrix, directed)[0]
    edge_total_line_graph_centr = np.sum(edge_line_graph_centr)
    return edge_total_line_graph_centr
# endregion

# region: Communication in Complex Networks
def total_network_sensitivity(adj_matrix, directed=False):
    '''Calculates the total network sensitivity of all edges and virtual edges in a network.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
            directed: boolean
                specifies if network is directed or undirected
        Returns:
            total_network_sensitivity_vector_downdating: ndarray
                the values of total network sensitivity for all edges of the network
            total_network_sensitivity_vector_updating: ndarray
                the values of total network sensitivity for all virtual edges of the network
            edge_list_downdating: list
                the edge list of all edges of the network
            edge_list_updating: list
                the edge list of all virtual edges of the network'''
    n = adj_matrix.shape[0]
    total_network_sensitivity_matrix = np.full((n, n), np.nan)
    E = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            E[i,j] = 1
            frechet_derivative = expm_frechet(adj_matrix, E, compute_expm=False)
            total_network_sensitivity_matrix[i,j] = (1 + directed) * np.sum(frechet_derivative)
            total_network_sensitivity_matrix[j,i] = (1 + directed) * np.sum(frechet_derivative)
            E[i,j] = 0

    tmp_adj_matrix_downdating = np.where(adj_matrix != 0, 1, np.nan)
    tmp_adj_matrix_updating = np.where(((adj_matrix == 0) & ~np.eye(adj_matrix.shape[0], dtype=bool)), 1, np.nan)
    total_network_sensitivity_matrix_downdating = np.multiply(total_network_sensitivity_matrix, tmp_adj_matrix_downdating)
    total_network_sensitivity_matrix_updating = np.multiply(total_network_sensitivity_matrix, tmp_adj_matrix_updating)

    if directed:
        total_network_sensitivity_vector_downdating = total_network_sensitivity_matrix_downdating.flatten()
        total_network_sensitivity_vector_updating = total_network_sensitivity_matrix_updating.flatten()
    else:
        total_network_sensitivity_vector_downdating = total_network_sensitivity_matrix_downdating[np.triu_indices(n)]
        total_network_sensitivity_vector_updating = total_network_sensitivity_matrix_updating[np.triu_indices(n)]

    total_network_sensitivity_vector_downdating = total_network_sensitivity_vector_downdating[~np.isnan(total_network_sensitivity_vector_downdating)]
    total_network_sensitivity_vector_updating = total_network_sensitivity_vector_updating[~np.isnan(total_network_sensitivity_vector_updating)]

    adj_matrix_edge_list_downdating = adj_matrix.copy()
    adj_matrix_edge_list_updating = np.where(((adj_matrix == 0) & ~np.eye(adj_matrix.shape[0], dtype=bool)), 1, 0)
    edge_list_downdating = tools.generate_edge_list(adj_matrix_edge_list_downdating, directed)
    edge_list_updating = tools.generate_edge_list(adj_matrix_edge_list_updating, directed)

    return total_network_sensitivity_vector_downdating, total_network_sensitivity_vector_updating, edge_list_downdating, edge_list_updating

def perron_root_sensitivity(adj_matrix, directed=False):
    '''Calculates the perron root sensitivity of all edges and virtual edges in a network.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
            directed: boolean
                specifies if network is directed or undirected
        Returns:
            perron_root_sensitivity_vector_downdating: ndarray
                the values of perron root sensitivity for all edges of the network
            perron_root_sensitivity_vector_updating: ndarray
                the values of perron root sensitivity for all virtual edges of the network
            edge_list_downdating: list
                the edge list of all edges of the network
            edge_list_updating: list
                the edge list of all virtual edges of the network'''
    # pertubate adj_matrix if adj_matrix is reducible
    if not tools.connectivity(adj_matrix, directed)[2]:
        # TODO: calculate delta with respect to the edge ranking
        delta = 0.00001
        A = adj_matrix + delta * np.ones(adj_matrix.shape)
        perron_root, perron_vector_right, perron_vector_left = tools.perron_root_and_eigenvectors(A)
    else:
        perron_root, perron_vector_right, perron_vector_left = tools.perron_root_and_eigenvectors(adj_matrix)

    perron_vector_right = perron_vector_right.reshape(perron_vector_right.size, 1)
    perron_vector_left = perron_vector_left.reshape(perron_vector_left.size, 1)

    perron_root_sensitivity_matrix = np.dot(perron_vector_left, perron_vector_right.T) / np.dot(perron_vector_left.T, perron_vector_right)

    tmp_adj_matrix_downdating = np.where(adj_matrix != 0, 1, np.nan)
    tmp_adj_matrix_updating = np.where(((adj_matrix == 0) & ~np.eye(adj_matrix.shape[0], dtype=bool)), 1, np.nan)

    perron_root_sensitivity_matrix_downdating = np.multiply(perron_root_sensitivity_matrix, tmp_adj_matrix_downdating)
    perron_root_sensitivity_matrix_updating = np.multiply(perron_root_sensitivity_matrix, tmp_adj_matrix_updating)
    if directed:
        perron_root_sensitivity_vector_downdating = perron_root_sensitivity_matrix_downdating.flatten()
        perron_root_sensitivity_vector_updating = perron_root_sensitivity_matrix_updating.flatten()
    else:
        perron_root_sensitivity_vector_downdating = perron_root_sensitivity_matrix_downdating[np.triu_indices(adj_matrix.shape[0])]
        perron_root_sensitivity_vector_updating = perron_root_sensitivity_matrix_updating[np.triu_indices(adj_matrix.shape[0])]

    perron_root_sensitivity_vector_downdating = perron_root_sensitivity_vector_downdating[~np.isnan(perron_root_sensitivity_vector_downdating)]
    perron_root_sensitivity_vector_updating = perron_root_sensitivity_vector_updating[~np.isnan(perron_root_sensitivity_vector_updating)]
    
    adj_matrix_edge_list_downdating = adj_matrix.copy()
    adj_matrix_edge_list_updating = np.where(((adj_matrix == 0) & ~np.eye(adj_matrix.shape[0], dtype=bool)), 1, 0)
    edge_list_downdating = tools.generate_edge_list(adj_matrix_edge_list_downdating, directed)
    edge_list_updating = tools.generate_edge_list(adj_matrix_edge_list_updating, directed)

    return perron_root_sensitivity_vector_downdating, perron_root_sensitivity_vector_updating, edge_list_downdating, edge_list_updating

def perron_network_communicability(adj_matrix):
    '''Calculates the perron network communicability of a network.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
        Returns:
            perron_network_comm: float
                the perron network communicability of the network'''
    perron_root, perron_vector_right, perron_vector_left = tools.perron_root_and_eigenvectors(adj_matrix)
    perron_network_comm = (np.exp(perron_root) - 1) * np.sum(perron_vector_right) * np.sum(perron_vector_left)
    return perron_network_comm

def perron_network_sensitivity(adj_matrix, directed=False):
    '''Calculates the perron network sensitivity of all edges and virtual edges in a network.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
            directed: boolean
                specifies if network is directed or undirected
        Returns:
            perron_network_sensitivity_vector_downdating: ndarray
                the values of perron network sensitivity for all edges of the network
            perron_network_sensitivity_vector_updating: ndarray
                the values of perron network sensitivity for all virtual edges of the network
            edge_list_downdating: list
                the edge list of all edges of the network
            edge_list_updating: list
                the edge list of all virtual edges of the network'''
    n = adj_matrix.shape[0]
    perron_network_sensitivity_matrix = np.full((n, n), np.nan)
    E = np.zeros((n, n))

    # pertubate adj_matrix if adj_matrix is reducible
    if not tools.connectivity(adj_matrix, directed)[2]:
        # TODO: calculate delta with respect to the edge ranking
        delta = 0.00001
        A = adj_matrix + delta * np.ones((n, n))
    else:
        A = adj_matrix.copy()

    perron_network_comm = perron_network_communicability(A)
    t = 0.00002
    for i in range(n):
        for j in range(n):
            E[i,j] = 1
            A_plus_E = A + t * E
            perron_network_comm_plus = perron_network_communicability(A_plus_E)
            perron_network_sensitivity_matrix[i,j] = (1 + directed) * 1/t * abs(perron_network_comm_plus - perron_network_comm)
            perron_network_sensitivity_matrix[j,i] = (1 + directed) * 1/t * abs(perron_network_comm_plus - perron_network_comm)
            E[i,j] = 0
            
    tmp_adj_matrix_downdating = np.where(adj_matrix != 0, 1, np.nan)
    tmp_adj_matrix_updating = np.where(((adj_matrix == 0) & ~np.eye(adj_matrix.shape[0], dtype=bool)), 1, np.nan)
    perron_network_sensitivity_matrix_downdating = np.multiply(perron_network_sensitivity_matrix, tmp_adj_matrix_downdating)
    perron_network_sensitivity_matrix_updating = np.multiply(perron_network_sensitivity_matrix, tmp_adj_matrix_updating)

    if directed:
        perron_network_sensitivity_vector_downdating = perron_network_sensitivity_matrix_downdating.flatten()
        perron_network_sensitivity_vector_updating = perron_network_sensitivity_matrix_updating.flatten()
    else:
        perron_network_sensitivity_vector_downdating = perron_network_sensitivity_matrix_downdating[np.triu_indices(n)]
        perron_network_sensitivity_vector_updating = perron_network_sensitivity_matrix_updating[np.triu_indices(n)]

    perron_network_sensitivity_vector_downdating = perron_network_sensitivity_vector_downdating[~np.isnan(perron_network_sensitivity_vector_downdating)]
    perron_network_sensitivity_vector_updating = perron_network_sensitivity_vector_updating[~np.isnan(perron_network_sensitivity_vector_updating)]

    adj_matrix_edge_list_downdating = adj_matrix.copy()
    adj_matrix_edge_list_updating = np.where(((adj_matrix == 0) & ~np.eye(adj_matrix.shape[0], dtype=bool)), 1, 0)
    edge_list_downdating = tools.generate_edge_list(adj_matrix_edge_list_downdating, directed)
    edge_list_updating = tools.generate_edge_list(adj_matrix_edge_list_updating, directed)

    return perron_network_sensitivity_vector_downdating, perron_network_sensitivity_vector_updating, edge_list_downdating, edge_list_updating
# endregion

# region: Sensitivity of matrix function based network communicability measures
def total_network_sensitivity_schweitzer(adj_matrix, directed=False):
    '''Calculates the total network sensitivity of all edges and virtual edges in a network.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
            directed: boolean
                specifies if network is directed or undirected
        Returns:
            total_network_sensitivity_vector_downdating: ndarray
                the values of total network sensitivity for all edges of the network
            total_network_sensitivity_vector_updating: ndarray
                the values of total network sensitivity for all virtual edges of the network
            edge_list_downdating: list
                the edge list of all edges of the network
            edge_list_updating: list
                the edge list of all virtual edges of the network'''
    n = adj_matrix.shape[0]
    II = np.ones((n, n))
    total_network_sensitivity_matrix = expm_frechet(adj_matrix.T, II.T, compute_expm=False)

    tmp_adj_matrix_downdating = np.where(adj_matrix != 0, 1, np.nan)
    tmp_adj_matrix_updating = np.where(((adj_matrix == 0) & ~np.eye(adj_matrix.shape[0], dtype=bool)), 1, np.nan)

    total_network_sensitivity_matrix_downdating = np.multiply(total_network_sensitivity_matrix, tmp_adj_matrix_downdating)
    total_network_sensitivity_matrix_updating = np.multiply(total_network_sensitivity_matrix, tmp_adj_matrix_updating)
    if directed:
        total_network_sensitivity_vector_downdating = total_network_sensitivity_matrix_downdating.flatten()
        total_network_sensitivity_vector_updating = total_network_sensitivity_matrix_updating.flatten()
    else:
        total_network_sensitivity_vector_downdating = total_network_sensitivity_matrix_downdating[np.triu_indices(adj_matrix.shape[0])]
        total_network_sensitivity_vector_updating = total_network_sensitivity_matrix_updating[np.triu_indices(adj_matrix.shape[0])]

    total_network_sensitivity_vector_downdating = total_network_sensitivity_vector_downdating[~np.isnan(total_network_sensitivity_vector_downdating)]
    total_network_sensitivity_vector_updating = total_network_sensitivity_vector_updating[~np.isnan(total_network_sensitivity_vector_updating)]
        
    adj_matrix_edge_list_downdating = adj_matrix.copy()  
    adj_matrix_edge_list_updating = np.where(((adj_matrix == 0) & ~np.eye(adj_matrix.shape[0], dtype=bool)), 1, 0)  
    edge_list_downdating = tools.generate_edge_list(adj_matrix_edge_list_downdating, directed)
    edge_list_updating = tools.generate_edge_list(adj_matrix_edge_list_updating, directed)
    
    return total_network_sensitivity_vector_downdating, total_network_sensitivity_vector_updating, edge_list_downdating, edge_list_updating
# endregion