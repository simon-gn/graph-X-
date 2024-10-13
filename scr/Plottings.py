import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import Tools as tools
import seaborn as sns

sns.set_theme(
    'paper',
    style='ticks',
    font='Lato',
    rc={
        'figure.figsize': [9, 7.2],
        'figure.edgecolor': '.3',
        'font.size': 30,
        'text.color': '.3',
        'axes.edgecolor': '.3',
        'axes.labelcolor': '.3',
        'axes.titlecolor': '.3',
        'axes.titleweight': 'bold',
        'grid.color': '.3',
        'xtick.color': '.3',
        'ytick.color': '.3',
        'text.usetex': False,
    }
)

def create_plot_network(adj_matrix, directed=False, edge_ranking=np.array([]), layout='spring', node_position={}, width_coding=False, color_coding=False, percentage_displayed_edges=100.0, percentage_color_coding=100, lowest_first=False, color='blue', node_size=250, line_width=1, edge_labeling='none', font_size_edge_labels=12, draw_node_labels=False, font_size_node_labels=12):
    '''Creates a plot of the network with customized options and node positions.
        Parameter:
            tabview: CTkTabview
                the tabview for the network plot
            tab_name: string
                the name of the tab for the network plot
            adj_matrix: ndarray
                the adjacency matrix of the network
            directed: boolean
                specifies if network is directed or undirected
            edge_ranking: list
                list of edge centrality values for all edges of the network determined by a specific edge centrality measure
            layout: string
                specifies the layout of the network
            node_position: list
                specifies the custom node positions
            width_coding: boolean
                specifies if width coding is enabled
            color_coding: boolean
                specifies if width coding is enabled
            percentage_displayed_edges: float
                specifies the amount of edges to display
            percentage_color_coding: float
                specifies the amount of edges to color
            lowest_first: boolean
                specifies the order of width or color coding
            color: string
                specifies the color of edges
            node_size: float
                specifies the size of the nodes
            line_width: float
                specifies the width of the edges
            edge_labeling: string
                specifies the edge labels
            font_size_edge_labels: float
                specifies the font size of the edge labels
            draw_node_labels: boolean
                specifies to draw node labels
            font_size_node_labels: float
                specifies the font size of the node labels'''
    # Create a graph object from the adjacency matrix
    if directed:
        G = nx.DiGraph(adj_matrix)
    else:
        G = nx.Graph(adj_matrix)
    G = nx.convert_node_labels_to_integers(G, 1, 'default', True)   # for node numbering starting at 1

    # Define the layout of the graph
    match layout:
        case 'spring':
            pos = nx.spring_layout(G)
        case 'circular':
            pos = nx.circular_layout(G)
        case 'spiral':
            pos = nx.spiral_layout(G)
        case 'planar':
            pos = nx.planar_layout(G)
        case 'spectral':
            pos = nx.spectral_layout(G)
        case 'shell':
            pos = nx.shell_layout(G)
        case 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        case 'random':
            pos = nx.random_layout(G)
    # Set predefined node positions
    for node_pos in node_position:
        pos[node_pos] = node_position[node_pos]

    # Label edges
    edge_labels = {}
    match edge_labeling:
        case 'numbering':
            for i, edge in enumerate(G.edges, start=1):
                edge_labels[edge] = i
        case 'coordinates':
            for edge in G.edges:
                edge_labels[edge] = edge
        case 'none':
            pass

    # Only display the X% highest or lowest edges
    if percentage_displayed_edges != 100.0:
        if not np.any(edge_ranking):
            # TODO: pop-up window
            raise ValueError('No edge ranking is given. An edge ranking is required for displaying a subset of edges.')
        below_threshold = tools.define_threshold(edge_ranking, percentage_displayed_edges, lowest_first)
        indices_of_removed_edges = []
        for i, edge in enumerate(G.edges):
            if below_threshold(edge_ranking[i]):
                G.remove_edge(*edge)
                if edge_labels:
                    edge_labels.pop(edge)
                indices_of_removed_edges.append(i)                
        edge_ranking = np.delete(edge_ranking, indices_of_removed_edges)

    # Color coding
    match color:
        case 'blue':
            cmap = plt.cm.Blues
        case 'red':
            cmap = plt.cm.Reds
        case 'gray':
            cmap = plt.cm.Greys
    # Color coding the X% highest or lowest edges
    if percentage_color_coding != 100:
        if not np.any(edge_ranking):
            # TODO: pop-up window
            raise ValueError('No edge ranking is given. An edge ranking is required for color-coding a subset of edges.')
        below_threshold =  tools.define_threshold(edge_ranking, percentage_color_coding, lowest_first)
        if color_coding:
            edges_to_color = [i for i in range(len(G.edges)) if not below_threshold(edge_ranking[i])]
            edge_centralities_to_color = edge_ranking[edges_to_color]
            edge_centralities_to_color_scaled = (tools.scale_data(edge_centralities_to_color, 0, 1)).tolist()
            edge_ranking_for_color_coding = []
            for i in range(len(G.edges)):
                if i not in edges_to_color:
                    edge_ranking_for_color_coding.append(edge_ranking[i])
                else:
                    edge_ranking_for_color_coding.append(edge_centralities_to_color_scaled.pop(0))
            edge_colors = ['gray' if below_threshold(edge_ranking[i]) else cmap(edge_ranking_for_color_coding[i]) for i in range(len(G.edges))]
        else:
            edge_colors = ['gray' if below_threshold(edge_ranking[i]) else color for i in range(len(G.edges))]
    else:
        edge_colors = 'gray'
    # Color coding all edges
        if color_coding:
            if not np.any(edge_ranking):
                # TODO: pop-up window
                raise ValueError('No edge ranking is given. An edge ranking is required for color-coding edges.')
            edge_colors = cmap(tools.scale_data(edge_ranking, 0, 1))

    # Width coding
    edge_widths = line_width
    if width_coding:
        if not np.any(edge_ranking):
            # TODO: pop-up window
            raise ValueError('No edge ranking is given. An edge ranking is required for width-coding edges.')
        edge_widths = tools.scale_data(edge_ranking, 0.05, line_width)

    # Draw the graph
    fig, ax = plt.subplots(1,1,figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=node_size)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths)
    if draw_node_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size_node_labels, font_family='sans-serif')
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=font_size_edge_labels, font_family='sans-serif')
    plt.axis('off')    
    plt.rcParams['savefig.dpi'] = 600
    plt.tight_layout()
    return fig

def create_plot_process(number_of_iterations, total_network_communicabilities_list, measure_names, selected_measures, order, iteration_disconnect, downdating):    
    '''Creates a plot of the updating or downdating results.
        Parameter:
            tabview: CTkTabview
                the tabview for the plot
            tab_name: list
                the names of the tabs for the plot
            number_of_iterations: int
                specifies the number of iterations of the process
            total_network_communicabilities_list: list
                list of lists of the values for the total network communicability in each iteration for each edge centrality measure
            measure_names: list
                the names of the selected measures
            selected_measures: list
                specifies the selected measures
            order: string
                specifies the order in which the network was down- or updated
            iteration_disconnect: list
                list of lists of the iterations in which the number of connected components increases for each edge centrality measure
            downdating: boolean
                specifies the type of process (downdating or updating)'''
    fig, ax = plt.subplots(1,1,figsize=(10, 8))
    markers = ['o','^','d','+','x']
    markevery = int(np.ceil(number_of_iterations/20))
    for i, total_network_comm in enumerate(total_network_communicabilities_list):
        ax.plot(total_network_comm, label=measure_names[selected_measures[i]], marker=markers[i], markevery=markevery, markeredgewidth=1, linewidth=1, mfc='none')
    
    if downdating and order == 'highest':
        colors = ['C0','C1','C2','C3','C4']
        for i in range(len(selected_measures)):
            if len(iteration_disconnect[i]):
                ax.axvline(iteration_disconnect[i][0], 0, 0, color=colors[i], marker=markers[i], markeredgewidth=1, mfc='none', zorder=10, clip_on=False)

    if downdating:
        ax.set_xlabel('Number of removed edges')
    else:
        ax.set_xlabel('Number of added edges')
    ax.set_ylabel('Total network communicability')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sns.despine(ax=ax, trim=True, offset=0)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig

def create_plot_correlation(ranking1, ranking2, xlabel, ylabel, filename, tau):
    '''Creates a plot of the correlation for the edge rankings of two edge centrality measures.
        Parameter:
            tabview: CTkTabview
                the tabview for the plot
            tab_name: list
                the names of the tabs for the plot
            ranking1: list
                the edge ranking of the first centrality measure
            ranking2: list
                the edge ranking of the second centrality measure
            xlabel: string
                the name of the first measures
            ylabel: string
                the name of the second measures
            filename: string
                the name of the file of the saved plot      
            tau: float
                the Kendall rank correlation coefficient for the given correlation
    '''
    fig, ax = plt.subplots(1,1,figsize=(10, 8))
    ax.plot(ranking1, ranking2, 'o', markersize=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    tau_math = r'$\tau$'
    ax.text(0.03, 0.92, f"{tau_math} = {tau}", ha='left', va='bottom', size='small', transform=plt.gca().transAxes)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sns.despine(ax=ax, trim=True, offset=0)
    plt.tight_layout()
    return fig

def create_plot_rankings(edge_rankings, measure_names, selected_measures):
    '''Creates a plot of the rankings of the selected edge centrality measures.
        Parameter:
            tabview: CTkTabview
                the tabview for the plot
            tab_name: list
                the names of the tabs for the plot
            edge_rankings: list
                list of lists of edge centrality values for all edges of the network for each selected centrality measure
            measure_names: list
                the names of the selected measures
            selected_measures: list
                specifies the selected measures'''
    fig, ax = plt.subplots(1,1,figsize=(10, 8))
    for i, ranking in enumerate(edge_rankings):
        ax.plot(tools.scale_data(ranking, 0, 1), label=measure_names[selected_measures[i]])

    ax.set_xlabel('Edges')
    ax.set_ylabel('Centrality Value')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sns.despine(ax=ax, trim=True, offset=5)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig

def create_plot_histogram(edge_rankings, measure_names, selected_measures):
    '''Creates a plot of the distribution of the edge rankings of the selected centrality measures.
        Parameter:
            tabview: CTkTabview
                the tabview for the plot
            tab_name: list
                the names of the tabs for the plot
            edge_rankings: list
                list of lists of edge centrality values for all edges of the network for each selected centrality measure
            measure_names: list
                the names of the selected measures
            selected_measures: list
                specifies the selected measures'''
    for i in range(len(edge_rankings)):
        edge_rankings[i] = tools.scale_data(edge_rankings[i], 0, 1)
    
    fig, ax = plt.subplots(1,1,figsize=(10, 8))
    ax.hist(edge_rankings, 20, label=[measure_names[selected_measures[i]] for i in range(len(edge_rankings))])

    ax.set_xlabel('Edge Centrality Value')
    ax.set_ylabel('Absolute Frequency')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sns.despine(ax=ax, trim=True, offset=5)
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig