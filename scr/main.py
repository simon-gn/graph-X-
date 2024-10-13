import sys
import tkinter as tk
from typing import Any, Optional, Tuple, Union
import customtkinter as ctk
import scipy.io as sio
import os
import numpy as np
import scipy.sparse as ssp
from scipy.linalg import issymmetric
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.backends.backend_pdf
import json
import NetworkAnalysis as na
import Tools as tools

class Custom_entry(ctk.CTkEntry):
    """A class of a CTkEntry with custom settings.
        Attributes:
            master: Any
                the parent window
            var: Any
                the textvariable of the CTkEntry
            row: int
                specifies the row in the grid to position the combobox
            state: Literal
                the initial state of the entry
            sticky: string
                specifies the side to which the entry should be stuck ("nesw" = (north, east, south, west))"""
    def __init__(self, master, var, row, state=ctk.DISABLED, sticky=''):
        ctk.CTkEntry.__init__(self, master, textvariable=var, state=state, text_color='gray')
        self.grid(row=row, column=1, padx=5, pady=5, sticky=sticky)

class Custom_combobox(ctk.CTkComboBox):
    """A class of a CTkComboBox with custom settings.
        Attributes:
            master: Any
                the parent window
            values: list
                the values to display in the combobox
            var: Any
                the variable of the CTkComboBox
            row: int
                specifies the row in the grid to position the combobox
            state: Literal
                the initial state of the combobox
            sticky: string
                specifies the side to which the combobox should be stuck ("nesw" = (north, east, south, west))"""
    def __init__(self, master, values, var, row, state=ctk.DISABLED, sticky=''):
        ctk.CTkComboBox.__init__(self, master, values=values, variable=var, state=state)
        self.set(var.get())
        self.grid(row=row, column=1, padx=5, pady=5, sticky=sticky)

class Custom_frame(ctk.CTkFrame):
    """A class of a CTkFrame with custom settings.
        Attributes:
            master: Any
                the parent window
            row: int
                specifies the row in the grid to position the frame
            column: int
                specifies the column in the grid to position the frame
            padx: int or tuple
                adds padding in x-direction
            pady: int or tuple
                adds padding in y-direction
            sticky: string
                specifies the side to which the frame should be stuck ("nesw" = (north, east, south, west))
            border_width: float
                specifies the width of the border surrounding the frame
            rowspan: int
                specifies to span the frame across multiple rows in the grid
            with_label: boolean
                determines to put a title on the top of the frame
            label_text: string
                the text of the title"""
    def __init__(self, master, row, column, padx=0, pady=0, sticky="nesw", border_width=0, rowspan=1, with_label=False, label_text=""):
        super().__init__(master=master, border_width=border_width)
        self.grid(row=row, column=column, rowspan=rowspan, padx=padx, pady=pady, sticky=sticky)
        if with_label:
            frame_label = ctk.CTkLabel(self, text=label_text, font=("Helvetica", 14, 'bold'))
            frame_label.pack(pady=(2,0))

class Custom_window(ctk.CTkToplevel):
    """A class of a CTkToplevel with custom settings.
        Attributes:
            master: Any
                the parent window
            title: string
                the title of the CTkToplevel
            size: string
                the size of the CtkToplevel"""
    def __init__(self, master, title, size):
        super().__init__(master)
        self.title(title)
        self.geometry(size)

class Pop_up_window(Custom_window):
    """A class for a custom pop-up window.
        Attributes:
            master: Any
                the parent window
            title: string
                the title of the window
            size: string
                the size of the window
            button_text: string
                the text of the button
            with_button: boolean
                determines to put a button on the bottom of the window
            label_text: string
                the text to display in the window"""
    def __init__(self, master, title, size, button_text="Okay", with_button=True, label_text=""):
        super().__init__(master, title, size)
        self.grab_set()
        label = ctk.CTkLabel(self, wraplength=300, justify=ctk.LEFT, text=label_text)
        label.pack(pady=(30,0))
        if with_button:
            confirm_button = ctk.CTkButton(self, text=button_text, command=self.destroy)
            confirm_button.pack(side=ctk.BOTTOM, pady=(0, 30))

class Option_window(Custom_window):
    """The class for the option window of Graph(X).
        Attributes:
            master: Any
                the parent window
            title: string
                the title of the window
            size: string
                the size of the window

        Methods:
            store_position():
                Saves the entered position of the entered node in the dictionary 'node_position' of NetworkApp(),
                if the position and the node are valid."""
    def __init__(self, master, title, size):
        super().__init__(master, title, size)
        tabview = ctk.CTkTabview(self)
        tab_plot_network_options = tabview.add("Plot network")
        tab_downdating_options = tabview.add("Downdating")
        tab_updating_options = tabview.add("Updating")
        tabview.pack(padx=20, pady=20)
        # initialize option values
        self._checkbox_plotting_options = [ctk.StringVar(value='spring'), ctk.BooleanVar(value=False), ctk.BooleanVar(value=False), 
                                          ctk.StringVar(value='100'), ctk.StringVar(value='100'), ctk.BooleanVar(value=False), 
                                          ctk.StringVar(value='blue'), ctk.StringVar(value='250'), ctk.StringVar(value='1'),
                                          ctk.StringVar(value='none'), ctk.StringVar(value='12'), ctk.BooleanVar(value=False), 
                                          ctk.StringVar(value='12')]
        self._checkbox_downdating_options = [ctk.StringVar(value='0'), ctk.BooleanVar(value=False), ctk.StringVar(value='lowest')]
        self._checkbox_updating_options = [ctk.StringVar(value='10'), ctk.BooleanVar(value=False), ctk.StringVar(value='highest')]
        self._node_positions = {}
        self._valid_node_inputs = []
        # first tab: plotting options
        plotting_options = ["Change layout:", "Enable width coding", "Enable color coding", "Only display X% of edges:", "Color only X% of edges:", "Lowest edges first", "Change color:", "Change node size:", "Change line width:", "Draw edge labels:", "Change font size (edge labels):", "Draw node labels", "Change font size (node labels):"]
        for i, option in enumerate(plotting_options):
            match option:
                case 'Change layout:':
                    layouts = ['spring', 'circular', 'spiral', 'spectral', 'shell', 'kamada_kawai', 'random']
                    combobox_layouts = Custom_combobox(master=tab_plot_network_options, values=layouts, var=self._checkbox_plotting_options[i], row=i)
                    checkbox = ctk.CTkCheckBox(master=tab_plot_network_options, text=option, command=lambda: tools.toggle_button_state(combobox_layouts, state='readonly'))
                # case 'Enable width coding':
                    # measures = ["Edge total communicability centrality", "Edge line graph centrality", "Total network sensitivity", "Perron root sensitivity", "Perron network sensitivity"]
                    # combobox_width_coding = custom_combobox(master=tab_plot_network_options, values=measures, var=self.app.checkbox_plotting_options[i], row=i)
                    # checkbox = ctk.CTkCheckBox(master=tab_plot_network_options, text=option, command=lambda: tools.toggle_button_state(combobox_width_coding, state='readonly'))
                # case 'Enable color coding':
                    # measures = ["Edge total communicability centrality", "Edge line graph centrality", "Total network sensitivity", "Perron root sensitivity", "Perron network sensitivity"]
                    # combobox_color_coding = custom_combobox(master=tab_plot_network_options, values=measures, var=self.app.checkbox_plotting_options[i], row=i)
                    # checkbox = ctk.CTkCheckBox(master=tab_plot_network_options, text=option, command=lambda: tools.toggle_button_state(combobox_color_coding, state='readonly'))
                case 'Only display X% of edges:':
                    entry_displayX = Custom_entry(master=tab_plot_network_options, var=self._checkbox_plotting_options[i], row=i)
                    checkbox = ctk.CTkCheckBox(master=tab_plot_network_options, text=option, command=lambda: tools.toggle_button_state(entry_displayX))
                case 'Color only X% of edges:':
                    entry_colorX = Custom_entry(master=tab_plot_network_options, var=self._checkbox_plotting_options[i], row=i)
                    checkbox = ctk.CTkCheckBox(master=tab_plot_network_options, text=option, command=lambda: tools.toggle_button_state(entry_colorX))
                case 'Change color:':
                    colors = ['blue', 'red', 'gray']
                    combobox_colors = Custom_combobox(master=tab_plot_network_options, values=colors, var=self._checkbox_plotting_options[i], row=i)
                    checkbox = ctk.CTkCheckBox(master=tab_plot_network_options, text=option, command=lambda: tools.toggle_button_state(combobox_colors, state='readonly'))
                case 'Change node size:':
                    entry_node_size = Custom_entry(master=tab_plot_network_options, var=self._checkbox_plotting_options[i], row=i)
                    checkbox = ctk.CTkCheckBox(master=tab_plot_network_options, text=option, command=lambda: tools.toggle_button_state(entry_node_size))
                case 'Change line width:':
                    entry_line_width = Custom_entry(master=tab_plot_network_options, var=self._checkbox_plotting_options[i], row=i)
                    checkbox = ctk.CTkCheckBox(master=tab_plot_network_options, text=option, command=lambda: tools.toggle_button_state(entry_line_width))
                case 'Draw edge labels:':
                    edge_labels = ['none', 'coordinates', 'numbering']
                    combobox_edge_labels = Custom_combobox(master=tab_plot_network_options, values=edge_labels, var=self._checkbox_plotting_options[i], row=i)
                    checkbox = ctk.CTkCheckBox(master=tab_plot_network_options, text=option, command=lambda: tools.toggle_button_state(combobox_edge_labels, state='readonly'))
                case 'Change font size (edge labels):':
                    entry_font_edge = Custom_entry(master=tab_plot_network_options, var=self._checkbox_plotting_options[i], row=i)
                    checkbox = ctk.CTkCheckBox(master=tab_plot_network_options, text=option, command=lambda: tools.toggle_button_state(entry_font_edge))
                case 'Change font size (node labels):':
                    entry_font_node = Custom_entry(master=tab_plot_network_options, var=self._checkbox_plotting_options[i], row=i)
                    checkbox = ctk.CTkCheckBox(master=tab_plot_network_options, text=option, command=lambda: tools.toggle_button_state(entry_font_node))
                case _:
                    checkbox = ctk.CTkCheckBox(master=tab_plot_network_options, text=option, variable=self._checkbox_plotting_options[i])
            checkbox.grid(row=i, column=0, padx=5, pady=5, sticky="w")
        # Set custom node positions TODO: revert to default functionality
        section_node_pos = ctk.CTkFrame(tab_plot_network_options, border_width=2)
        section_node_pos.grid(row=13, column=0, columnspan=2, pady=30, sticky="nesw")
        self._var_node_pos = ctk.StringVar(value="node 1")
        self._combobox_node_list = Custom_combobox(master=section_node_pos, values=self._valid_node_inputs, var=self._var_node_pos, row=0, sticky='e')
        self._entry_xPos = Custom_entry(master=section_node_pos, var="", row=1, sticky='e')
        self._entry_yPos = Custom_entry(master=section_node_pos, var="", row=2, sticky='e')
        label_xPos = ctk.CTkLabel(master=section_node_pos, text="x:", state=ctk.DISABLED)
        label_xPos.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        label_yPos = ctk.CTkLabel(master=section_node_pos, text="y:", state=ctk.DISABLED)
        label_yPos.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        button_save_pos = ctk.CTkButton(master=section_node_pos, text="Save Position", command=self.store_position, state=ctk.DISABLED)
        button_save_pos.grid(row=3, column=1, padx=(33.5,5), pady=5, sticky="w")
        checkbox = ctk.CTkCheckBox(master=section_node_pos, text='Set custom node position:', command=lambda: [tools.toggle_button_state(self._combobox_node_list), tools.toggle_button_state(self._entry_xPos), tools.toggle_button_state(self._entry_yPos), tools.toggle_button_state(label_xPos), tools.toggle_button_state(label_yPos), tools.toggle_button_state(button_save_pos, text_color=('#DCE4EE', '#DCE4EE'))])
        checkbox.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        # second tab: downdating options
        downdating_options = ["Change number of iterations:", "Enable greedy mode", "Change order:"]
        for i, option in enumerate(downdating_options):
            match option:
                case 'Change number of iterations:':
                    entry_number_iterations_d = Custom_entry(master=tab_downdating_options, var=self._checkbox_downdating_options[i], row=i)
                    checkbox = ctk.CTkCheckBox(master=tab_downdating_options, text=option, command=lambda: tools.toggle_button_state(entry_number_iterations_d))
                case 'Change order:':
                    order = ['lowest', 'highest']
                    combobox_order_d = Custom_combobox(master=tab_downdating_options, values=order, var=self._checkbox_downdating_options[i], row=i)
                    checkbox = ctk.CTkCheckBox(master=tab_downdating_options, text=option, command=lambda: tools.toggle_button_state(combobox_order_d, state='readonly'))
                case _:
                    checkbox = ctk.CTkCheckBox(master=tab_downdating_options, text=option, variable=self._checkbox_downdating_options[i])
            checkbox.grid(row=i, column=0, padx=5, pady=5, sticky="w")
        # third tab: updating options
        updating_options = ["Change number of iterations:", "Enable greedy mode", "Change order:"]
        for i, option in enumerate(updating_options):
            match option:
                case 'Change number of iterations:':
                    entry_number_iterations_u = Custom_entry(master=tab_updating_options, var=self._checkbox_updating_options[i], row=i)
                    checkbox = ctk.CTkCheckBox(master=tab_updating_options, text=option, command=lambda: tools.toggle_button_state(entry_number_iterations_u))
                case 'Change order:':
                    order = ['highest', 'lowest']
                    combobox_order_u = Custom_combobox(master=tab_updating_options, values=order, var=self._checkbox_updating_options[i], row=i)
                    checkbox = ctk.CTkCheckBox(master=tab_updating_options, text=option, command=lambda: tools.toggle_button_state(combobox_order_u, state='readonly'))
                case _:
                    checkbox = ctk.CTkCheckBox(master=tab_updating_options, text=option, variable=self._checkbox_updating_options[i])
            checkbox.grid(row=i, column=0, padx=5, pady=5, sticky="w")
        confirm_button = ctk.CTkButton(self, text='Confirm', command=self.withdraw)
        confirm_button.pack(side=ctk.BOTTOM, pady=(0, 30))
        self.withdraw()
        self.protocol("WM_DELETE_WINDOW", self.withdraw)    # change behavior when window is closed via the x-button

    def store_position(self):
        if self._var_node_pos.get() in self._valid_node_inputs:
            node_nr = int(self._var_node_pos.get().split()[1])
            xPos = self._entry_xPos.get()
            yPos = self._entry_yPos.get()
            try:
                pos = (float(xPos), float(yPos))
                self._node_positions[node_nr] = pos
                Pop_up_window(self, "Task completed", "300x150", label_text="Node position saved successfully!")
            except ValueError:
                Pop_up_window(self, "Error", "300x150", label_text="Invalid position. Please enter a valid position.")
        else:
            Pop_up_window(self, "Error", "300x150", label_text="Invalid node. Please enter a valid node.")

    def update_valid_nodes(self, valid_node_inputs):
        self._combobox_node_list.configure(values=valid_node_inputs)
        self._valid_node_inputs = valid_node_inputs
    
    def get_user_input_options(self):
        return self._checkbox_plotting_options, self._checkbox_downdating_options, self._checkbox_updating_options, self._node_positions
    
    def set_user_input_options(self, plotting_options_input, downdating_options_input, updating_options_input, node_positions, valid_node_inputs):
        for i in range(len(self._checkbox_plotting_options)):
            self._checkbox_plotting_options[i].set(plotting_options_input[i])
        for i in range(len(self._checkbox_downdating_options)):
            self._checkbox_downdating_options[i].set(downdating_options_input[i])
        for i in range(len(self._checkbox_updating_options)):
            self._checkbox_updating_options[i].set(updating_options_input[i])
        self._node_positions = node_positions
        self.update_valid_nodes(valid_node_inputs)

class Section_load_network(Custom_frame):
    """The class for the frame which contains the 'Open File' button, the text field for the network filename and the 'directed' checkbox in the panel.
        Attributes:
            master: Any
                the parent window
            row: int
                specifies the row in the grid to position the frame
            column: int
                specifies the column in the grid to position the frame
            padx: int or tuple
                adds padding in x-direction
            pady: int or tuple
                adds padding in y-direction
            border_width: float
                specifies the width of the border surrounding the frame
        
        Methods:
            display_network_filename(filename):
                Creates a text field for the network filename and displays the name."""
    def __init__(self, master, row, column, padx=10, pady=(0,20), border_width=2):
        super().__init__(master=master, row=row, column=column, padx=padx, pady=pady, border_width=border_width)
        self._listeners = []
        self._directed = ctk.BooleanVar(value=False)
        button_open_file = ctk.CTkButton(master=self, text="Open File", command=self.notify)
        button_open_file.grid(row=0, column=0, padx=5, pady=7, sticky="w")
        checkbox = ctk.CTkCheckBox(master=self, text="directed", variable=self._directed)
        checkbox.grid(row=1, column=0, padx=5, pady=(3,5), sticky="w")

    def attach(self, listener):
        self._listeners.append(listener)

    def notify(self):
        for listener in self._listeners:
            listener.update("load network file")

    def set_directed(self, val):
        self._directed.set(val)

    def display_network_filename(self, filename):
        filename_textbox = ctk.CTkTextbox(master=self, height=1, width=130, fg_color='white', activate_scrollbars=False, text_color='gray10')
        filename_textbox.grid(row=0, column=1, padx=(4.5,5), pady=5, sticky='w')
        filename_textbox.insert(ctk.END, filename)
        filename_textbox.configure(state=ctk.DISABLED)

class Section_select_tasks(Custom_frame):
    """The class for the frame which contains the checkboxes for the tasks and the 'Customize Options' button in the panel.
        Attributes:
            master: Any
                the parent window
            row: int
                specifies the row in the grid to position the frame
            column: int
                specifies the column in the grid to position the frame
            padx: int or tuple
                adds padding in x-direction
            pady: int or tuple
                adds padding in y-direction
            border_width: float
                specifies the width of the border surrounding the frame"""
    def __init__(self, master, row, column, padx=10, pady=(0,20), border_width=2):
        super().__init__(master=master, row=row, column=column, padx=padx, pady=pady, border_width=border_width)
        self._listeners = []
        self._checkbox_tasks = [ctk.BooleanVar(value=False) for _ in range(6)]
        tasks = ["Plot network", "Downdate network", "Update network", "Plot ranking(s)", "Plot histogram", "Plot correlations"]
        for i, task in enumerate(tasks):
            checkbox = ctk.CTkCheckBox(master=self, text=task, variable=self._checkbox_tasks[i])
            if i < 3:
                checkbox.grid(row=i, column=0, padx=5, pady=5, sticky="w")
            else:
                checkbox.grid(row=i-3, column=1, padx=5, pady=5, sticky="w")    
        button_costumize_options = ctk.CTkButton(master=self, text="Customize Options", command=self.notify)
        button_costumize_options.grid(row=3, column=0, padx=5, pady=7, sticky="nsew")

    def attach(self, listener):
        self._listeners.append(listener)

    def notify(self):
        for listener in self._listeners:
            listener.update("customize options")

    def set_user_input_options(self, tasks_input):
        for i in range(len(self._checkbox_tasks)):
            self._checkbox_tasks[i].set(tasks_input[i])

class Section_pick_centrality_measures(Custom_frame):
    """The class for the frame which contains the checkboxes for the edge centrality measures in the panel.
        Attributes:
            master: Any
                the parent window
            row: int
                specifies the row in the grid to position the frame
            column: int
                specifies the column in the grid to position the frame
            padx: int or tuple
                adds padding in x-direction
            pady: int or tuple
                adds padding in y-direction
            border_width: float
                specifies the width of the border surrounding the frame"""
    def __init__(self, master, row, column, padx=10, pady=(0,20), border_width=2):
        super().__init__(master=master, row=row, column=column, padx=padx, pady=pady, border_width=border_width)
        self._checkbox_measures = [ctk.BooleanVar(value=False) for _ in range(5)]
        measures = ["Edge total communicability centrality", "Edge line graph centrality", "Total network sensitivity", "Perron root sensitivity", "Perron network sensitivity"]
        for i, measure in enumerate(measures):
            checkbox = ctk.CTkCheckBox(master=self, text=measure, variable=self._checkbox_measures[i])
            checkbox.grid(row=i, column=0, padx=5, pady=5, sticky="w")

    def set_user_input_options(self, measure_input):
        for i in range(len(self._checkbox_measures)):
            self._checkbox_measures[i].set(measure_input[i])

class Frame_settings(Custom_frame):
    """The class for the panel sub-window of Graph(X).
        Attributes:
            master: Any
                the parent window
            row: int
                specifies the row in the grid to position the frame
            column: int
                specifies the column in the grid to position the frame
            rowspan: int
                specifies to span the frame across multiple rows in the grid"""
    def __init__(self, master, row=0, column=0, rowspan=2):
        super().__init__(master=master, row=row, column=column, rowspan=rowspan)
        self._listeners = []
        # Create sections
        self.section_load_network = self.create_section("Load network data file:", Section_load_network, row=0, column=0, font_size=14, padx=15, pady=(5,0), sticky="w")
        self.section_select_tasks = self.create_section("Select task(s):", Section_select_tasks, row=2, column=0, font_size=14, padx=15, pady=(5,0), sticky="w")
        self.section_pick_centrality_measures = self.create_section("Pick Centrality Measure(s):", Section_pick_centrality_measures, row=4, column=0, font_size=14, padx=15, sticky="w")
        # Create Buttons
        self.create_button("Run", "run calculations", row=6, column=0, padx=(10,0), pady=5, sticky="w")
        self.create_button("Load", "load data", row=6, column=0, padx=(0,10), pady=5, sticky="e")
        self.create_button("Save Plots", "save plots", row=7, column=0, padx=10, pady=5, sticky="w")
        self.create_button("Save Data", "save data", row=7, column=0, padx=10, pady=5, sticky="e")
        self.create_button("Quit", "quit", row=8, column=0, padx=10, pady=5, sticky="nsew")

    def create_section(self, label_text, section_class, **kwargs):
        label = ctk.CTkLabel(self, text=label_text, font=('Helvetica', kwargs.get('font_size'), 'bold'))
        label.grid(row=kwargs.get('row'), column=kwargs.get('column'), padx=kwargs.get('padx'), pady=kwargs.get('pady'), sticky=kwargs.get('sticky'))
        section = section_class(master=self, row=kwargs.get('row') + 1, column=kwargs.get('column'))
        return section

    def create_button(self, text, event_name, **kwargs):
        button = ctk.CTkButton(master=self, text=text, command=lambda: self.notify(event_name))
        button.grid(**kwargs)

    def attach(self, listener):
        self._listeners.append(listener)
        self.section_load_network.attach(listener)
        self.section_select_tasks.attach(listener)

    def notify(self, event_name):
        for listener in self._listeners:
            listener.update(event_name)

    def get_user_input_options(self):
        return self.section_load_network._directed, self.section_select_tasks._checkbox_tasks, self.section_pick_centrality_measures._checkbox_measures

class Frame_network_visualization(Custom_frame):
    """The class for the sub-window 'Network visualization' of Graph(X).
        Attributes:
            master: Any
                the parent window
            row: int
                specifies the row in the grid to position the frame
            column: int
                specifies the column in the grid to position the frame
            border_width: float
                specifies the width of the border surrounding the frame
            with_label: boolean
                determines to put a title on the top of the frame
            label_text: string
                the text of the title

        Methods:
            manage_tabs(self, tasks_input):
                Adds a tab for the new network plot to the already existing tabview or creates a new tabiew with the tab if no tabview yet exists.
                Returns the tabview and the name of the new tab."""
    def __init__(self, master, row=0, column=1, border_width=0.5, with_label=True, label_text="Network visualization"):
        super().__init__(master=master, row=row, column=column, border_width=border_width, with_label=with_label, label_text=label_text)
        self._tabview = ctk.CTkTabview(self)

    def plot(self, figures):
        num_tabs = tools.get_number_of_tabs(self._tabview, "Network")
        tab_name = f"Network {num_tabs+1}"
        self._tabview.add(tab_name)
        # plot the figure
        canvas = FigureCanvasTkAgg(figures["Network"], self._tabview.tab(tab_name))
        canvas.draw()
        canvas.get_tk_widget().pack(pady=(0,10))
        self._tabview.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0,10))
        self._tabview.set(tab_name)

class Frame_plots(Custom_frame):
    """The class for the sub-window 'Plots' of Graph(X).
        Attributes:
            master: Any
                the parent window
            row: int
                specifies the row in the grid to position the frame
            column: int
                specifies the column in the grid to position the frame
            border_width: float
                specifies the width of the border surrounding the frame
            with_label: boolean
                determines to put a title on the top of the frame
            label_text: string
                the text of the title

        Methods:
            manage_tabs(self, tasks_input):
                Adds tabs for the plots specified by the given tasks to the already existing tabview or creates a new tabiew with tabs if no tabview yet exists.
                Returns the tabview and the new tabs with names as a dictionary."""
    def __init__(self, master, row=1, column=1, border_width=0.5, with_label=True, label_text="Plots"):
        super().__init__(master=master, row=row, column=column, border_width=border_width, with_label=with_label, label_text=label_text)
        self._tabview = ctk.CTkTabview(self)
        self._tab_names = ["Downdating", "Updating", "Rankings", "Histogram"]

    def plot(self, tasks_input, figures):
        for i in range(4):
            if tasks_input[i+1]:
                num_tabs = tools.get_number_of_tabs(self._tabview, self._tab_names[i])
                tab_name = f"{self._tab_names[i]} {num_tabs+1}"
                self._tabview.add(tab_name)
                # plot the figure
                canvas = FigureCanvasTkAgg(figures[self._tab_names[i]], self._tabview.tab(tab_name))
                canvas.draw()
                canvas.get_tk_widget().pack(pady=(0,10))
                self._tabview.set(tab_name)
        self._tabview.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0,10))

class Model:
    """The Model class of Graph(X).
        Methods:
            load_network_file():
                Loads the network data file and examines various properties of the network.
            
            run_calculations():
                Runs the selected tasks with the specified options.

            save_plots():
                Saves all plots to the Plots folder in the working directory.
            
            save_data():
                Saves data to a JSON-file in the working directory with filename 'data'. 
                The data comprises of the adjacency matrix, the network filename, 
                the last calculated edge rankings including the centrality values of all selected centrality measures, 
                the selected tasks, the selected options, the specified node positions.
            
            load_data():
                Loads the data of the 'data.json' file in the working directory.

            largest_connected_component(window):
                Gets the largest connected component of the loaded network.

            remove_self_loops(window):
                Removes the self-loops of the loaded network."""
    def __init__(self):
        # Initialize variables
        self._adj_matrix = np.empty(0)
        self._filename = ""
        self._edge_centralities_d, self._edge_centralities_u, self._ranked_edge_lists_d, self._ranked_edge_lists_u = [], [], [], []
        self._figures = {}
        self._listeners = []
        # Set default saving directory
        current_working_directory = os.getcwd()
        if not os.path.exists(current_working_directory):
            os.makedirs(current_working_directory)
        self._default_directory = os.path.abspath(os.path.join(current_working_directory, ".."))
    
    def attach(self, listener):
        self._listeners.append(listener)
    
    def notify(self, event_name, *args, **kwargs):
        for listener in self._listeners:
            listener.update(event_name, *args, **kwargs)

    def load_network_file(self):
        # Search for supported formats in the directory and load the network data if possible
        file_path = tk.filedialog.askopenfilename(filetypes=[("MATLAB files", "*.mat"), ("Scipy Sparce files", "*.npz")])
        if os.path.isfile(file_path):
            file_extension = os.path.splitext(file_path)[1]
            match file_extension:
                case ".mat":
                    data = sio.loadmat(file_path)
                    self._adj_matrix = data['Problem']['A'][0][0].todense()
                case ".npz":
                    data = ssp.load_npz(file_path)
                    self._adj_matrix = data.todense()
                case _:
                    print("File format is not supported! Supported formats are .mat and .nzp files")
                    return
        else:
            print("The file path does not exist.")
            return
        # Check for right array type
        if type(self._adj_matrix).__name__ != np.ndarray.__name__:
            self._adj_matrix = np.asarray(self._adj_matrix)
        # Check if network is directed or undirected
        directed = not issymmetric(self._adj_matrix)
        # Check if network has self-loops
        A_diag = np.diag(self._adj_matrix)
        if np.count_nonzero(A_diag) != 0:
            self.notify("warning pop up", size="350x190", text="The network contains self-loops. The edge centrality measures are not suited for networks with self-loops. Would you like to remove the self-loops?", func="remove self loops")
        # Check if network is (strongly) connected
        if not tools.connectivity(self._adj_matrix, directed)[2]:
            self.notify("warning pop up", size="350x160", text="The network is not connected. Would you like to only consider the largest connected component?", func="largest connected component", directed=directed)
        # Check if network is weighted
        if np.any((self._adj_matrix > 1) | ((self._adj_matrix < 1) & (self._adj_matrix != 0))):
            print('Graph is weighted!')
        # Check if the values of the adj_matrix are valid
        if np.any(self._adj_matrix != self._adj_matrix.astype(int)):
            self.notify("warning pop up", size="350x160", text="The values of some elements of the adjacency matrix are not integers. Some features will not work properly!")
        # List the nodes of the network
        valid_node_inputs = ['node ' + str(i) for i in range(1, self._adj_matrix.shape[0] + 1)]
        # Get name of the network file
        self._filename = os.path.basename(file_path)
        return valid_node_inputs, self._filename, directed

    def run_calculations(self, tasks_input, measure_input, plotting_options_input, downdating_options_input, updating_options_input, node_positions, directed):
        self._edge_centralities_d, self._edge_centralities_u, self._ranked_edge_lists_d, self._ranked_edge_lists_u = na.compute_centrality_values(self._adj_matrix, measure_input, directed)
        new_figures = na.run_tasks(self._adj_matrix, self._edge_centralities_d, self._edge_centralities_u, self._ranked_edge_lists_d, self._ranked_edge_lists_u, [i for i, measure in enumerate(measure_input) if measure], tasks_input, plotting_options_input, downdating_options_input, updating_options_input, node_positions, directed)
        self._figures = tools.merge_dicts(self._figures, new_figures)
        return new_figures

    def save_plots(self):
        '''Saves all plots currently present in the instance of Graph(X).'''
        os.makedirs(f'{self._default_directory}/plots', exist_ok=True)
        for filename, fig in self._figures.items():
            fig.savefig(f'{self._default_directory}/plots/{filename}', dpi=fig.dpi)
        self.notify("task completed", None, "Plots saved!")

    def save_data(self, tasks_input, measure_input, plotting_options_input, downdating_options_input, updating_options_input, node_positions, directed):
        data_to_save = {"A": [arr.tolist() for arr in self._adj_matrix],
                        "filename": self._filename,
                        "edge_centralities_d": [arr.tolist() for arr in self._edge_centralities_d], 
                        "edge_centralities_u": [arr.tolist() for arr in self._edge_centralities_u], 
                        "ranked_edge_lists_d": self._ranked_edge_lists_d, 
                        "ranked_edge_lists_u": self._ranked_edge_lists_u,
                        "tasks_input": tasks_input,
                        "measure_input": measure_input,
                        "plotting_options_input": plotting_options_input,
                        "downdating_options_input": downdating_options_input,
                        "updating_options_input": updating_options_input,
                        "node_positions": node_positions,
                        "directed": directed}
        serialized_data = json.dumps(data_to_save)
        file_path = os.path.join(self._default_directory, 'data.json')
        with open(file_path, "w") as file:
            file.write(serialized_data)
        self.notify("task completed", None, "Data saved!")

    def load_data(self):
        # try to load file
        file_path = os.path.join(self._default_directory, 'data.json')
        try:
            with open(file_path, "r") as file:
                loaded_data = json.load(file)
        except FileNotFoundError:
            print('No file found!')         # TODO: pop-up window
            return
        except json.JSONDecodeError:
            print('File is corrupted!')     # TODO: pop-up window
            return
        # load data of file
        A_to_list = loaded_data["A"]
        self._adj_matrix = np.array(A_to_list)
        valid_node_inputs = ['node ' + str(i) for i in range(1, self._adj_matrix.shape[0] + 1)]
        self._filename = loaded_data["filename"]
        edge_centralities_d_to_list = loaded_data["edge_centralities_d"]
        self._edge_centralities_d = [np.array(list) for list in edge_centralities_d_to_list]
        edge_centralities_u_to_list = loaded_data["edge_centralities_u"]
        self._edge_centralities_u = [np.array(list) for list in edge_centralities_u_to_list]
        ranked_edge_lists_d_to_list = loaded_data["ranked_edge_lists_d"]
        self._ranked_edge_lists_d = []
        for i in range(len(ranked_edge_lists_d_to_list)):
            self._ranked_edge_lists_d.append([tuple(edge) for edge in ranked_edge_lists_d_to_list[i]])
        ranked_edge_lists_u_to_list = loaded_data["ranked_edge_lists_u"]
        self._ranked_edge_lists_u = []
        for i in range(len(ranked_edge_lists_u_to_list)):
            self._ranked_edge_lists_u.append([tuple(edge) for edge in ranked_edge_lists_u_to_list[i]])
        tasks_input = loaded_data["tasks_input"]
        measure_input = loaded_data["measure_input"]
        plotting_options_input = loaded_data["plotting_options_input"]
        downdating_options_input = loaded_data["downdating_options_input"]
        updating_options_input = loaded_data["updating_options_input"]
        node_positions = loaded_data["node_positions"]
        directed = loaded_data["directed"]
        # create figures with loaded data and return data and figures
        self.notify("display network filename", self._filename)
        new_figures = na.run_tasks(self._adj_matrix, self._edge_centralities_d, self._edge_centralities_u, self._ranked_edge_lists_d, self._ranked_edge_lists_u, [i for i, measure in enumerate(measure_input) if measure], tasks_input, plotting_options_input, downdating_options_input, updating_options_input, node_positions, directed)
        self._figures = tools.merge_dicts(self._figures, new_figures)
        return tasks_input, measure_input, plotting_options_input, downdating_options_input, updating_options_input, node_positions, valid_node_inputs, directed, new_figures
    
    def largest_connected_component(self, directed):
        self._adj_matrix = tools.get_largest_connected_component(self._adj_matrix, directed)
        self.notify("task completed", "The network now only consists of the largest connected component!")

    def remove_self_loops(self):
        for i in range(self._adj_matrix.shape[0]):
            self._adj_matrix[i,i] = 0
        self.notify("task completed", "Self-loops removed successfully!")

class View:
    """The View class of Graph(X).
        Methods:
            display_user_interface():
                Executes the main loop.

            quit():
                Closes the instance of the application.
                
            warning_pop_up(warning_text, func):
                Creates a warning pop-up window of the given size with the given warning text and a 'Yes' and 'No' button.
                The pressing of the 'Yes' button triggers the given function."""
    def __init__(self):
        ctk.set_appearance_mode("System")
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Graph(X)")
        self.root.geometry("1100x1000")
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.notify("quit"))
        # Create option window
        self.option_window = Option_window(master=self.root, title="Options", size="450x900")
        # Create main frames in main window
        self.frame_settings = Frame_settings(master=self.root)
        self.frame_network_visualization = Frame_network_visualization(master=self.root)
        self.frame_plots = Frame_plots(master=self.root)
        # Window settings
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=50)
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.attributes('-topmost', False)
        # instantiate list for listeners
        self._listeners = []

    def attach(self, listener):
        self._listeners.append(listener)
        self.frame_settings.attach(listener)

    def notify(self, event_name, *args):
        for listener in self._listeners:
            listener.update(event_name, *args)
    
    def display_user_interface(self):
        self.root.mainloop()

    def get_user_input_options(self):
        return self.frame_settings.get_user_input_options(), self.option_window.get_user_input_options()
    
    def plot(self, tasks_input, figures):
        if tasks_input[0]:
            self.frame_network_visualization.plot(figures)
        if np.any(tasks_input[1:]):
            self.frame_plots.plot(tasks_input, figures)
    
    def warning_pop_up(self, size, text, func=None, directed=False):
        if func == None:
            warning_pop_up = Pop_up_window(self.root, "Warning", size, with_button=True, label_text=text)
        else:
            warning_pop_up = Pop_up_window(self.root, "Warning", size, with_button=False, label_text=text)
            button_yes = ctk.CTkButton(warning_pop_up, text='Yes', command=lambda: self.on_confirmation(func, directed, warning_pop_up))
            button_yes.pack(side=ctk.LEFT, padx=(30,0))
            button_no = ctk.CTkButton(warning_pop_up, text='No', command=warning_pop_up.destroy)
            button_no.pack(side=ctk.RIGHT, padx=(0,30))
        self.root.wait_window(warning_pop_up)
    
    def on_confirmation(self, event_name, directed, window):
        self.notify(event_name, directed)
        window.destroy()
    
    def task_completed_pop_up(self, text):
        window_task_completed = Pop_up_window(self.root, "Task completed", "300x150", with_button=False, label_text=text)
        button_confirm = ctk.CTkButton(window_task_completed, text="Okay", command=window_task_completed.destroy)
        button_confirm.pack(side=ctk.BOTTOM, pady=(0, 30))
        self.root.wait_window(window_task_completed)

class Controller:
    """The Controller class of Graph(X).
        Methods:
            update(event_name, *args):
                Triggers methods with optional parameters 'args' in the Model or View class with respect to the given event."""
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.model.attach(self)
        self.view.attach(self)
    
    def update(self, event_name, *args, **kwargs):
        match event_name:
            case "load network file":
                valid_node_inputs, filename, directed = self.model.load_network_file()
                self.view.frame_settings.section_load_network.display_network_filename(filename)
                self.view.frame_settings.section_load_network.set_directed(directed)
                self.view.option_window.update_valid_nodes(valid_node_inputs)
            case "run calculations":
                tasks_input, measure_input, plotting_options_input, downdating_options_input, updating_options_input, node_positions, directed = self.get_user_inputs_from_view()
                # run the calculations and create the plots
                figures = self.model.run_calculations(tasks_input, measure_input, plotting_options_input, downdating_options_input, updating_options_input, node_positions, directed)
                # display the plots
                self.view.plot(tasks_input, figures)
            case "save plots":
                self.model.save_plots()
            case "save data":
                tasks_input, measure_input, plotting_options_input, downdating_options_input, updating_options_input, node_positions, directed = self.get_user_inputs_from_view()
                self.model.save_data(tasks_input, measure_input, plotting_options_input, downdating_options_input, updating_options_input, node_positions, directed)
            case "load data":
                tasks_input, measure_input, plotting_options_input, downdating_options_input, updating_options_input, node_positions, valid_node_inputs, directed, figures = self.model.load_data()
                self.view.option_window.set_user_input_options(plotting_options_input, downdating_options_input, updating_options_input, node_positions, valid_node_inputs)
                self.view.frame_settings.section_select_tasks.set_user_input_options(tasks_input)
                self.view.frame_settings.section_pick_centrality_measures.set_user_input_options(measure_input)
                self.view.frame_settings.section_load_network.set_directed(directed)
                self.view.plot(tasks_input, figures)
            case "quit":
                self.quit()
            case "customize options":
                self.view.option_window.deiconify()
            case "warning pop up":
                self.view.warning_pop_up(**kwargs)
            case "remove self loops":
                self.model.remove_self_loops()
            case "largest connected component":
                self.model.largest_connected_component(*args)
            case "task completed":
                self.view.task_completed_pop_up(*args)
            case "display network filename":
                self.view.frame_settings.section_load_network.display_network_filename(*args)

    def get_user_inputs_from_view(self):
        options = self.view.get_user_input_options()
        directed, checkbox_tasks, checkbox_measures, checkbox_plotting_options, checkbox_downdating_options, checkbox_updating_options, node_positions = options[0] + options[1]
        tasks_input = [var.get() for var in checkbox_tasks]
        measure_input = [var.get() for var in checkbox_measures]
        plotting_options_input = [var.get() for var in checkbox_plotting_options]
        downdating_options_input = [var.get() for var in checkbox_downdating_options]
        updating_options_input = [var.get() for var in checkbox_updating_options]
        return tasks_input, measure_input, plotting_options_input, downdating_options_input, updating_options_input, node_positions, directed.get()
    
    def quit(self):
        try:
            self.view.root.destroy() # some pending events seems to be not finishing correctly
            # sys.exit()
        except Exception as e:
            print(f"An exception occurred: {e}")

class NetworkApp:
    """The main class of Graph(X). It instantiates the Model, View and Controller class. 
        Methods:
            run():
                Runs the app."""
    def __init__(self):
        self.model = Model()
        self.view = View()
        self.controller = Controller(self.model, self.view)
    
    def run(self):
        self.view.display_user_interface()

if __name__ == "__main__":
    app = NetworkApp()
    app.run()