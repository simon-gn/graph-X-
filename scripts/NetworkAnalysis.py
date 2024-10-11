import numpy as np
import EdgeCentralityMeasures as ecm
import Tools as tools
import Plottings as plts
from scipy.linalg import issymmetric
import copy
from enum import Enum
from scipy.stats import kendalltau

class Tasks(Enum):
    plot_network = 0
    downdating = 1
    updating = 2
    plot_rankings = 3
    plot_histogram = 4
    plot_correlations = 5

class Measures(Enum):
    edge_total_communicability_centrality = 0
    edge_line_graph_centrality = 1
    total_network_sensitivity = 2
    perron_root_sensitivity = 3
    perron_network_sensitivity = 4

class Plotting_options(Enum):
    layout = 0
    width_coding = 1
    color_coding = 2
    percentage_displayed_edges = 3
    percentage_color_coding = 4
    lowest_first = 5
    color = 6
    node_size = 7
    line_width = 8
    edge_labeling = 9
    font_size_edge_labels = 10
    draw_node_labels = 11
    font_size_node_labels = 12

class Downup_options(Enum):
    number_of_iterations = 0
    greedy = 1
    order = 2

measure_names = ['Edge total communicability centrality', 'Edge line graph centrality', 'Total network sensitivity', 'Perron root sensitivity', 'Perron network sensitivity']

def compute_centrality_values(adj_matrix, measure_input, directed):
    '''Calculates and returns the centrality of all edges and virtual edges of a network using specified edge centrality measures.
        Parameter:
            adj_matrix: ndarray
                the adjacency matrix of the network
            measure_input: list
                specifies the selected edge centrality measure
            directed: boolean
                specifies if network is directed or undirected
        Returns:
            edge_centralities_d: list
                list of lists of edge centrality values for each centrality measures
            edge_centralities_u: list
                list of lists of centrality values of all virtual edges for each centrality measures 
            ranked_edge_list_d: list
                list of edge lists ranked by the centrality value for each centrality measures
            ranked_edge_list_u: list
                list of virual edge lists ranked by the centrality value for each centrality measures'''
    if not directed and not issymmetric(adj_matrix):
        raise ValueError("The adjacency matrix of the graph is not symmetric. Either the graph is directed or the given data is missing some edges. If the Graph is directed ensure to select 'directed' in the options.")
    
    edge_centralities_d = []
    edge_centralities_u = []
    ranked_edge_lists_d = []
    ranked_edge_lists_u = []
    
    if measure_input[Measures.edge_total_communicability_centrality.value]:
        edge_total_communicability_centr_d, edge_total_communicability_centr_u, edge_list_d, edge_list_u = ecm.edge_total_communicability_centrality(adj_matrix, directed)
        ranked_values_d, ranked_edge_list_d = tools.rank_edges(edge_total_communicability_centr_d, edge_list_d, "Edge total communicability centrality:")
        ranked_values_u, ranked_edge_list_u = tools.rank_edges(edge_total_communicability_centr_u, edge_list_u, "Edge total communicability centrality:")
        edge_centralities_d.append(edge_total_communicability_centr_d)
        edge_centralities_u.append(edge_total_communicability_centr_u)
        ranked_edge_lists_d.append(ranked_edge_list_d)
        ranked_edge_lists_u.append(ranked_edge_list_u)

    if measure_input[Measures.edge_line_graph_centrality.value]:
        edge_line_graph_centr_d, edge_list_d = ecm.edge_line_graph_centrality(adj_matrix, directed)
        ranked_values_d, ranked_edge_list_d = tools.rank_edges(edge_line_graph_centr_d, edge_list_d, "Edge line graph centrality:")
        edge_centralities_d.append(edge_line_graph_centr_d)
        ranked_edge_lists_d.append(ranked_edge_list_d)

    if measure_input[Measures.total_network_sensitivity.value]:
        total_network_sens_d, total_network_sens_u, edge_list_d, edge_list_u = ecm.total_network_sensitivity_schweitzer(adj_matrix, directed)
        ranked_values_d, ranked_edge_list_d = tools.rank_edges(total_network_sens_d, edge_list_d, "Total network sensitivity:")
        ranked_values_u, ranked_edge_list_u = tools.rank_edges(total_network_sens_u, edge_list_u, "Total network sensitivity:")
        edge_centralities_d.append(total_network_sens_d)
        edge_centralities_u.append(total_network_sens_u)
        ranked_edge_lists_d.append(ranked_edge_list_d)
        ranked_edge_lists_u.append(ranked_edge_list_u)

    if measure_input[Measures.perron_root_sensitivity.value]:
        perron_root_sens_d, perron_root_sens_u, edge_list_d, edge_list_u = ecm.perron_root_sensitivity(adj_matrix, directed)
        ranked_values_d, ranked_edge_list_d = tools.rank_edges(perron_root_sens_d, edge_list_d, "Perron root sensitivity:")
        ranked_values_u, ranked_edge_list_u = tools.rank_edges(perron_root_sens_u, edge_list_u, "Perron root sensitivity:")
        edge_centralities_d.append(perron_root_sens_d)
        edge_centralities_u.append(perron_root_sens_u)
        ranked_edge_lists_d.append(ranked_edge_list_d)
        ranked_edge_lists_u.append(ranked_edge_list_u)

    if measure_input[Measures.perron_network_sensitivity.value]:
        perron_network_sens_d, perron_network_sens_u, edge_list_d, edge_list_u = ecm.perron_network_sensitivity(adj_matrix, directed)
        ranked_values_d, ranked_edge_list_d = tools.rank_edges(perron_network_sens_d, edge_list_d, "Perron network sensitivity:")
        ranked_values_u, ranked_edge_list_u = tools.rank_edges(perron_network_sens_u, edge_list_u, "Perron network sensitivity:")
        edge_centralities_d.append(perron_network_sens_d)
        edge_centralities_u.append(perron_network_sens_u)
        ranked_edge_lists_d.append(ranked_edge_list_d)
        ranked_edge_lists_u.append(ranked_edge_list_u)

    return edge_centralities_d, edge_centralities_u, ranked_edge_lists_d, ranked_edge_lists_u

def run_tasks(adj_matrix, edge_centralities_d, edge_centralities_u, ranked_edge_lists_d, ranked_edge_lists_u, selected_measures, tasks_input, plotting_options_input, downdating_options_input, updating_options_input, node_position, directed):
    '''Executes the corresponding methods of the selected tasks.
        Parameter:
            tabview_network: CTkTabview
                the tabview for the network plots
            tab_name_network: string
                name of the tabs for the network plots
            tabview_plots: CTkTabview
                the tabview for the other plots
            tab_names_plots: list
                names of the tabs for the other plots
            adj_matrix: ndarray
                the adjacency matrix of the network
            edge_centralities_d: list
                list of lists of edge centrality values for each centrality measures
            edge_centralities_u: list
                list of lists of centrality values of all virtual edges for each centrality measures 
            ranked_edge_list_d: list
                list of edge lists ranked by the centrality value for each centrality measures
            ranked_edge_list_u: list
                list of virual edge lists ranked by the centrality value for each centrality measures
            selected_measures: list
                specifies the selected edge centrality measures
            tasks_input: list
                specifies the selected tasks
            plotting_options_input: list
                specifies the selected options for plotting the network
            downdating_options_input: list
                specifies the selected options for downdating the network
            updating_options_input: list
                specifies the selected options for updating the network
            node_position: list
                specifies the custom node positions
            directed: boolean
                specifies if network is directed or undirected'''
    figures = {}
    if tasks_input[Tasks.plot_network.value]:
        if not edge_centralities_d:
            edge_centralities_d.append([])
        figures["Network"] = plts.create_plot_network(adj_matrix=adj_matrix, directed=directed, edge_ranking=edge_centralities_d[0], 
                          layout=plotting_options_input[Plotting_options.layout.value],
                          node_position=node_position,
                          width_coding=plotting_options_input[Plotting_options.width_coding.value], 
                          color_coding=plotting_options_input[Plotting_options.color_coding.value],
                          percentage_displayed_edges=float(plotting_options_input[Plotting_options.percentage_displayed_edges.value]),
                          percentage_color_coding=float(plotting_options_input[Plotting_options.percentage_color_coding.value]),
                          lowest_first=plotting_options_input[Plotting_options.lowest_first.value],
                          color=plotting_options_input[Plotting_options.color.value],
                          node_size=int(plotting_options_input[Plotting_options.node_size.value]),
                          line_width=float(plotting_options_input[Plotting_options.line_width.value]),
                          edge_labeling=plotting_options_input[Plotting_options.edge_labeling.value],
                          font_size_edge_labels=float(plotting_options_input[Plotting_options.font_size_edge_labels.value]),
                          draw_node_labels=plotting_options_input[Plotting_options.draw_node_labels.value],
                          font_size_node_labels=float(plotting_options_input[Plotting_options.font_size_node_labels.value]))

    if tasks_input[Tasks.downdating.value]:
        figures["Downdating"] = compare_measures(adj_matrix, ranked_edge_lists_d, selected_measures, downdating_options_input, directed)

    if tasks_input[Tasks.updating.value]:
        figures["Updating"] = compare_measures(adj_matrix, ranked_edge_lists_u, selected_measures, updating_options_input, directed, False)

    # TODO: choice for ploting rankings for virtual edges (edge_centralities_u)
    if tasks_input[Tasks.plot_rankings.value]:
        figures["Rankings"] = plts.create_plot_rankings(edge_centralities_d, measure_names, selected_measures)

    # TODO: choice for plotting histogram for virtual edges (edge_centralities_u)
    if tasks_input[Tasks.plot_histogram.value]:
        figures["Histogram"] = plts.create_plot_histogram(edge_centralities_d, measure_names, selected_measures)

    # TODO: choice for plotting correlations for virtual edges (edge_centralities_u)
    if tasks_input[Tasks.plot_correlations.value]:
        for i, centrality in enumerate(edge_centralities_d):
            tmp_edge_centralities = edge_centralities_d.copy()
            tmp_edge_centralities.pop(i)
            tmp_selected_measures = selected_measures.copy()
            tmp_selected_measures.pop(i)
            for j in range(i, len(tmp_edge_centralities)):
                tau = kendalltau(centrality, tmp_edge_centralities)[0]
                plts.create_plot_correlation(tools.scale_data(centrality, 0, 1), tools.scale_data(tmp_edge_centralities[j], 0, 1), measure_names[selected_measures[i]], measure_names[tmp_selected_measures[j]], f'{i+1}_{j+2}.pdf', tau)
    
    return figures

def compare_measures(adj_matrix, ranked_edge_lists, selected_measures, downup_options_input, directed, downdating=True):
    '''Compares the selected measure using the downdating or updating process.
        Parameter:
            tabview_plots: CTkTabview
                the tabview for the other plots
            tab_names: list
                names of the tabs for the other plots
            adj_matrix: ndarray
                the adjacency matrix of the network
            ranked_edge_list: list
                list of edge lists ranked by the centrality value for each centrality measures (list of virtual edges when updating the network)
            selected_measures: list
                specifies the selected edge centrality measures
            downup_options_input: list
                specifies the selected options for downdating or updating the network
            downdating: boolean
                specifies if network should be downdated or updated'''
    ranked_edge_lists_copy = copy.deepcopy(ranked_edge_lists)
    adj_matrices_modified = [adj_matrix.copy() for _ in range(len(ranked_edge_lists_copy))]

    if int(downup_options_input[Downup_options.number_of_iterations.value]) == 0:
        number_of_iterations = int(np.sum(adj_matrix))
    else:
        number_of_iterations = int(downup_options_input[Downup_options.number_of_iterations.value])

    n_components = [tools.connectivity(adj_matrix, directed)[0] for _ in range(len(ranked_edge_lists_copy))] 
    iterations_disconnect = [[] for _ in range(len(ranked_edge_lists_copy))]
    total_network_comm_init = tools.total_network_communicability(adj_matrix)
    total_network_communicabilities_list = [[total_network_comm_init] for _ in range(len(ranked_edge_lists_copy))]
    
    print("Total Network Communicability:")
    print("Before downdating:", total_network_comm_init)
    print()

    if not directed:
        number_of_iterations //= 2
    for k in range(number_of_iterations):
        modified_edges = []
        for i, ranked_edges in enumerate(ranked_edge_lists_copy):
            if downup_options_input[Downup_options.order.value] == 'lowest':
                edge_modified = ranked_edges[-1]
            elif downup_options_input[Downup_options.order.value] == 'highest':
                edge_modified = ranked_edges[0]
            else:
                raise ValueError("Invalid order parameter: should be 'lowest' or 'highest'")
            modified_edges.append(edge_modified)

            if downdating:
                adj_matrices_modified[i] = tools.downdate_network(adj_matrices_modified[i], edge_modified, directed)
                # da die Kantengewichte immer bloß um 1 gesenkt werden, kann die last_edge erst von der ranked_edges Liste entfernt werden, wenn das Kantengewicht auf 0 gefallen ist
                if adj_matrices_modified[i][edge_modified] == 0:
                    ranked_edges.remove(edge_modified)
                    if n_components[i] != tools.connectivity(adj_matrices_modified[i], directed)[0]:
                        iterations_disconnect[i].append(k+1)
                        n_components[i] = tools.connectivity(adj_matrices_modified[i], directed)[0]
                if downup_options_input[Downup_options.greedy.value]:
                    new_measure_input = [0] * 5
                    new_measure_input[selected_measures[i]] = True
                    ranked_edge_lists_copy[i] = compute_centrality_values(adj_matrices_modified[i], new_measure_input, directed)[2][0]

            else:
                adj_matrices_modified[i] = tools.update_network(adj_matrices_modified[i], edge_modified, directed)
                ranked_edges.remove(edge_modified)
                if downup_options_input[Downup_options.greedy.value]:
                    new_measure_input = [0] * 5
                    new_measure_input[selected_measures[i]] = True
                    ranked_edge_lists_copy[i] = compute_centrality_values(adj_matrices_modified[i], new_measure_input, directed)[3][0]

        print(f"Iteration {k+1}:")
        total_network_communicabilities = []
        for adj_matrix_modified in adj_matrices_modified:
            total_network_communicabilities.append(tools.total_network_communicability(adj_matrix_modified))
        
        print("Modified Edges:", modified_edges)
        print("Total Network Communicability:")
        for i, total_network_comm in enumerate(total_network_communicabilities):
            print(f"{measure_names[selected_measures[i]]}: {total_network_comm}")
            total_network_communicabilities_list[i].append(total_network_comm)
        print()

    return plts.create_plot_process(number_of_iterations, total_network_communicabilities_list, measure_names, selected_measures, downup_options_input[Downup_options.order.value], iterations_disconnect, downdating)