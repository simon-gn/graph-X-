import numpy as np
import networkx as nx

def generate_line_graph(adj_matrix, directed=False):
    '''
    Generates the line graph of a given graph.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the graph
            directed: boolean
                specifies if graph is directed or undirected
        Returns:
            L: Graph
                the line graph
            E: ndarray
                the adjacency matrix of the line graph
    '''
    
    if directed:
        G = nx.DiGraph(adj_matrix)
    else:
        G = nx.Graph(adj_matrix)

    G = nx.convert_node_labels_to_integers(G, 1, 'default', True)

    G_line = nx.line_graph(G)

    # Bring the nodes in the correct order
    L = nx.Graph()
    L.add_nodes_from(sorted(G_line.nodes))
    for u, v in sorted(G_line.edges):
        if G_line.has_edge(u, v):
            L.add_edge(u, v)
    
    # Create the line graph adjacency matrix
    E = nx.adjacency_matrix(L)
    E = np.array(E.todense(), dtype=float)

    return L, E

def line_graph_adj_matrix(adj_matrix):
    '''
    Returns the adjacency matrix of the line graph to a given undirected graph.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the undirected graph
        Returns:
            E: ndarray
                the adjacency matrix of the line graph
    '''
    
    B = incidence_matrix(adj_matrix)
    E = B.T @ B
    for i in range(E.shape[0]):
        E[i,i] = 0

    return E

def line_graph_adj_matrix_directed(adj_matrix):
    '''
    Returns the adjacency matrix of the line graph to a given directed graph.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the directed graph
        Returns:
            E: ndarray
                the adjacency matrix of the line graph
    '''
    
    B_i, B_e = incidence_and_exsurgence_matrix(adj_matrix)    
    E = B_i.T @ B_e

    return E

def incidence_matrix(adj_matrix):
    '''
    Returns the incidence matrix of an undirected graph.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the undirected graph
        Returns:
            inc_matrix: ndarray
                the incidence matrix of the undirected graph
    '''
    
    n = adj_matrix.shape[0]
    edges = []
    weights = []
    for i in range(n):
        for j in range(i, n):
            if(adj_matrix[i,j] != 0):
                edges.append((i,j))
                weights.append(adj_matrix[i,j])
                
    inc_matrix = np.zeros((n, len(edges)))
    for i, edge in enumerate(edges):
        inc_matrix[edge, i] = np.sqrt(weights[i])

    return inc_matrix

def incidence_and_exsurgence_matrix(adj_matrix):
    '''
    Returns the incidence and exsurgence matrix of a directed graph.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the directed graph
        Returns:
            inc_matrix: ndarray
                the incidence matrix of the directed graph
            exsur_matrix: ndarray
                the exsurgence matrix of the directed graph
    '''
    
    n = adj_matrix.shape[0]
    heads = []
    tails = []
    weights = []
    for i in range(n):
        for j in range(n):
            if adj_matrix[i,j] != 0:
                heads.append(j)
                tails.append(i)
                weights.append(adj_matrix[i,j])
    
    inc_matrix = np.zeros((n, len(heads)))
    exsur_matrix = np.zeros((n, len(tails)))
    for i, (head, tail) in enumerate(zip(heads, tails)):
        sqrt_weight = np.sqrt(weights[i])
        inc_matrix[head, i] = sqrt_weight
        exsur_matrix[tail, i] = sqrt_weight

    return inc_matrix, exsur_matrix