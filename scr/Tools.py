import numpy as np
import scipy.sparse as ssp
from scipy.linalg import expm, eig
import customtkinter as ctk

def total_network_communicability(adj_matrix):
    '''Calculates the total network communicability of a network.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
        Returns:
            total_comm: float
                the total network communicability of the network'''
    exp_adj = expm(adj_matrix)
    total_comm = np.sum(exp_adj)

    return total_comm

def rank_edges(data, edge_list, measurement):
    '''Sorts the edges of a network by the centrality values of a specific edge centrality measure in descending order.
        Parameter:
            data: ndarray
                the centrality values of a specific edge centrality measures
            edge_list: list
                the edge list of the network
            measurement: string
                the name of the edge centrality measure
        Returns:
            ranked_data: list
                the sorted edge centrality values
            ranked_edge_list: list
                the sorted edge list'''
    ranked_data = data.copy()
    ranked_data[::-1].sort()
    sorted_indices = np.argsort(-data)     # -data for descending order
    ranked_edge_list = []
    for i in sorted_indices:
        ranked_edge_list.append(edge_list[i])
    
    # print the list of edges with corresponding centrality values to the console
    """print('\t', "Edge Nr.", '\t', "edge:", '\t', '\t', measurement)
    for i, value in enumerate(ranked_data):
        print(i+1, '\t', sorted_indices[i] + 1, '\t', '\t', ranked_edge_list[i], '\t', value)
    print()"""
   
    return ranked_data, ranked_edge_list

def downdate_network(adj_matrix, edge, directed=False):
    '''Removes an edge of a given network.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
            edge: tuple
                the edge of the network to remove
            directed: boolean
                specifies if network is directed or undirected
        Returns:
            adj_matrix_downdated: ndarray
                the downdated adjacency matrix with the edge removed'''
    adj_matrix_downdated = adj_matrix.copy()

    if adj_matrix[edge] < 1:
        adj_matrix_downdated[edge] = 0
        if not directed:
            adj_matrix_downdated[edge[1], edge[0]] = 0
    else:
        adj_matrix_downdated[edge] -= 1
        if not directed:
            adj_matrix_downdated[edge[1], edge[0]] -= 1

    return adj_matrix_downdated

def update_network(adj_matrix, edge, directed=False):
    '''Adds an edge to a given network.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
            edge: tuple
                the edge to add to the network
            directed: boolean
                specifies if network is directed or undirected
        Returns:
            adj_matrix_updated: ndarray
                the updated adjacency matrix with the edge added'''
    adj_matrix_updated = adj_matrix.copy()

    adj_matrix_updated[edge] += 1
    if not directed:
        adj_matrix_updated[edge[1], edge[0]] += 1

    return adj_matrix_updated

def scale_data(data, new_min, new_max):
    '''Scales the given data to the interval from new_min to new_max.
        Parameter:
            data: list or ndarray
                the data to scale
            new_min: float
                the new minimum of the scaled data
            new_max: float
                the new maximum of the scaled data
        Returns:
            data_scaled: ndarray
                the scaled data'''
    max_value = max(data)
    min_value = min(data)
    if max_value != min_value:
        data_scaled = [((x - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min  for x in data]
    else:
        data_scaled = [0.5 * (new_max + new_min)] * len(data)

    data_scaled = np.asarray(data_scaled)
    return data_scaled

def get_largest_connected_component(adj_matrix, directed):
    '''Returns the largest connected component of a given network.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
            directed: boolean
                specifies if network is directed or undirected
        Returns:
            A: ndarray
                the adjacency matrix of the largest connected component of the given network'''
    n_components, labels, _ = connectivity(adj_matrix, directed=directed)
    n = 0
    for i in range(n_components):
        x = np.count_nonzero(labels == i)
        if x > n:
            n = x
            label = i
    largest_con_comp = [i for i, element in enumerate(labels) if element == label]
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i,j] = adj_matrix[largest_con_comp[i], largest_con_comp[j]]
    
    return A

def connectivity(adj_matrix, directed):
    '''Analyzes the connectivity of a given network.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
            directed: boolean
                specifies if network is directed or undirected
        Returns:
            n_components: int
                the number of connected components
            labels: ndarray
                the length-N array of labels of the connected components
            is_connected: boolean
                specifies if the network is (strongly) connected'''
    sparse_matrix = ssp.csr_matrix(adj_matrix)
    n_components, labels = ssp.csgraph.connected_components(sparse_matrix, directed=directed, connection='strong')
    is_connected = (n_components == 1)
    return n_components, labels, is_connected

def generate_edge_list(adj_matrix, directed):
    '''Returns the edge list of a given network.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
            directed: boolean
                specifies if network is directed or undirected
        Returns:
            edge_list: list
                the edge list of the given network'''
    if directed:
        row_indices, col_indices = adj_matrix.nonzero()
    else:
        row_indices, col_indices = ssp.triu(adj_matrix).nonzero()
        
    row_indices = [int(index) for index in row_indices]
    col_indices = [int(index) for index in col_indices]
        
    edge_list = list(zip(row_indices, col_indices))

    return edge_list

def perron_root_and_eigenvectors(matrix):
    '''Calculates the Perron root and the Perron eigenvectors of a matrix.
        Parameter:
            matrix: ndarray
                the matrix to calculate the Perron root and Perron eigenvectors
        Returns:
            perron_root: float
                the Perron root of the matrix
            perron_vector_right: ndarray
                the right Perron vector of the matrix
            perron_vector_left: ndarray
                the left Perron vector of the matrix'''
    eigvals, eigvecs_left, eigvecs_right = eig(matrix, left=True) # might consider using eigs() for better performance
    # find the index of the Perron root (the eigenvalue with the largest real part)
    perron_index = np.argmax(np.real(eigvals))
    perron_root = np.real(eigvals[perron_index])

    # get the Perron vectors corresponding to the Perron root
    perron_vector_right = np.real(eigvecs_right[:, perron_index])
    perron_vector_left = np.real(eigvecs_left[:, perron_index])
    
    if(perron_vector_right[0] < 0):
        perron_vector_right *= -1
        perron_vector_left *= -1

    return perron_root, perron_vector_right, perron_vector_left

def sum_rows_and_reshape(matrix):
    '''Calculates the sum of each row of a matrix.
        Parameter:
            matrix: ndarray
                the matrix to calculate the sum of the rows
        Returns:
            row_sum: ndarray
                the sum of each row'''
    row_sum = np.sum(matrix, axis=1)
    row_sum = row_sum.reshape(row_sum.size, 1)
    return row_sum

def define_threshold(edge_ranking, percentage, lowest_first):
    '''Returns a lambda function to check if the centrality value of a specific edge is above or below a certain threshold.
        Parameter:
            edge_ranking: ndarray
                the centrality values of the edges
            percentage: float
                determines the threshold (the amount of edges which should satisfy the lambda function)
            lowest_first: boolean
                determines if the percentage specification refers to the edges with the highest or lowest values
        Returns:
            condition: lambda function
                checks if the input x is above or below the threshold, returns a boolean'''
    if not lowest_first:
        condition = lambda x: x < value_threshold
        percentage = 100.0 - percentage
    else:
        condition = lambda x: x > value_threshold
    index_threshold = int(np.ceil((percentage/100 * len(edge_ranking)))) - 1
    value_threshold = np.sort(edge_ranking)[index_threshold]

    return condition

def toggle_button_state(button, state=ctk.NORMAL, text_color=('gray10', '#DCE4EE')):
    '''Toggles the state of a button/checkbox/entry/combobox.
        Parameters:
            button: CTkButton/CTkCheckbox/CTkEntry/CTkCombobox
                the button/checkbox/entry/combobox to toggle the state
            state: Literal
                the state to toggle to
            text_color: tuple
                the text_color when the button/checkbox/... is not in disabled state ('light', 'dark')'''
    if button.cget('state') == ctk.DISABLED:
        button.configure(state=state, text_color=text_color)
    else:
        button.configure(state=ctk.DISABLED, text_color='gray')
    
def get_number_of_tabs(tabview, tab_name):
    '''Returns the number of tabs with the name 'tab_name' in a tabview.
        Parameters:
            tabview: CTkTabview
                the tabview to check
            tab_name: string
                the name of the tab to count
        Returns:
            num_tabs: int
                the number of tabs'''
    num_tabs = 0
    while True:
        try:
            tabview.tab(f"{tab_name} {num_tabs+1}")
            num_tabs += 1
        except ValueError:
            break
    return num_tabs

def merge_dicts(dict1, dict2):
    result = dict1.copy()
    duplicate_counter = 2  # Initialize a counter for duplicate keys

    for key, value in dict2.items():
        while key in result:  # Check if the key already exists in the result dictionary
            key = f"{key}{duplicate_counter}"  # Append a number to the key
            duplicate_counter += 1
        result[key] = value  # Add the key-value pair to the result dictionary

    return result