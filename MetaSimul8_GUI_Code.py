#Name: Sindhiya Sridharan
#Student Id: 201629979 


# This is MetaSimul8 UI code : Kinldy use this to work with MetaSimul8
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from PIL import Image, ImageTk
from MetaSimul8_Main_Simulation_code import (convert_matrix_to_dict, best_response_against, calculate_new_frequencies, calculate_expected_payoff,calculate_win_percentage, calculate_overall_win_percentages, update_payoff_matrix, add_new_strategy, adjust_remaining_frequencies,Strategy_Popularity, update_population_with_changes, simulate_Metasimul8, display_graphical_representation_dynamic_v2, pacman_dominance,show_strategy_plotter)
from tkinter import PhotoImage, Label

original_payoff_matrix = []
filtered_matrix = []
original_strategy_names = []
filtered_strategy_names = []
strategy_frequency = []
selected_strategies = []

def create_matrix(num_strategies, values):
    # Create a matrix based on the user inputs
    matrix = []
    for row in values:
        matrix.append([float(val.strip()) for val in row.split(',')])
    return matrix

def create_table(matrix, strategy_names):
    # Create a pandas DataFrame from the matrix with custom row and column names
    df = pd.DataFrame(matrix, index=strategy_names, columns=strategy_names)
    return df

def filter_data(matrix, strategy_names):
    global filtered_strategy_names
    selected_strategies = selected_strategies_listbox.curselection()
    if selected_strategies:
        selected_strategies = [strategy_names[index] for index in selected_strategies]
        filtered_strategy_names = selected_strategies  # Store the filtered strategy names separately
        filtered_matrix = [[matrix[strategy_names.index(i)][strategy_names.index(j)] for j in selected_strategies] for i in selected_strategies]
        return filtered_matrix
    else:
        return None

def create_matrix_table():
    global matrix, strategy_names, original_payoff_matrix, original_strategy_names, original_matrix

    try:
        num_strategies = int(num_strategies_entry.get())
        if num_strategies < 1:
            raise ValueError("Number of strategies should be a positive integer.")

        strategy_names = strategy_names_entry.get().split(',')
        strategy_names = [name.strip() for name in strategy_names]
        if len(strategy_names) != num_strategies:
            raise ValueError("Number of strategy names should match the number of strategies.")

        values = matrix_values_entry.get("1.0", tk.END).strip().split('\n')
        if len(values) != num_strategies:
            raise ValueError(f"Number of rows in values should be equal to the number of strategies ({num_strategies}).")
        for row in values:
            if len(row.split(',')) != num_strategies:
                raise ValueError(f"Number of values in each row should be equal to the number of strategies ({num_strategies}).")
    except ValueError as e:
        messagebox.showerror("Error", str(e))
        return

    matrix = create_matrix(num_strategies, values)
    original_matrix = matrix  # Store the original matrix separately
    original_payoff_matrix = matrix
    original_strategy_names = strategy_names

    table = create_table(matrix, strategy_names)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, str(table))

def filter_data_gui():
    global filtered_matrix, filtered_strategy_names, strategy_frequency
    if not original_matrix:
        messagebox.showerror("Error", "Matrix not created yet. Please create a matrix first.")
        filter_var.set(False)
        return

    filter_choice = filter_var.get()
    if filter_choice:
        selected_strategies_var.set('')
        selected_strategies_listbox.delete(0, tk.END)
        selected_strategies_listbox.insert(tk.END, *strategy_names)
    else:
        selected_strategies_listbox.delete(0, tk.END)
        selected_strategies_var.set('')
        filtered_matrix = None
        filtered_strategy_names = []
        strategy_frequency = []
        output_text.delete(1.0, tk.END)


def apply_filter():
    global filtered_matrix
    filtered_matrix = filter_data(original_matrix, original_strategy_names)
    if filtered_matrix:
        table = create_table(filtered_matrix, filtered_strategy_names)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, str(table))
    else:
        messagebox.showinfo("Info", "No valid strategy names selected. Filtered data table cannot be created.")

def on_exit():
    if messagebox.askyesno("Exit", "Do you want to exit?"):
        root.destroy()


strategy_freq_dict = {}

def update_strategy_frequency_option(*args):
    selected_frequency_option = strategy_frequency_var.get()
    if selected_frequency_option == "Manual":
        value_label.grid(row=9, column=0, padx=5, pady=5)
        value_entry.grid(row=9, column=1, padx=5, pady=5)
    else:
        value_label.grid_forget()
        value_entry.grid_forget()

def calculate_strategy_frequency():
    global strategy_frequency
    global strategy_freq_dict
    
    strategy_freq_dict = {}
    strategy_frequency.clear()

    population_size = float(population_size_entry.get())
    selected_frequency_option = strategy_frequency_var.get()

    # Determine whether to use the filtered strategies or the original ones
    if filter_var.get():
        strategies_to_use = [original_strategy_names[index] for index in selected_strategies_listbox.curselection()]
    else:
        strategies_to_use = original_strategy_names

    if population_size and len(strategies_to_use) > 0:
        if selected_frequency_option == "Equal Division":
            frequency = population_size / len(strategies_to_use)
            normalized_frequency = frequency / population_size
            output_text.delete(1.0, tk.END)
            output_text.insert(tk.END, f"Equal Division: {normalized_frequency:.3f}")
            strategy_frequency.append((selected_frequency_option, normalized_frequency))

            # Assign normalized frequency to each strategy in strategy_freq_dict
            for strategy in strategies_to_use:
                strategy_freq_dict[strategy] = normalized_frequency
            
        elif selected_frequency_option == "Manual":
            values = value_entry.get().split(',')
            if len(values) == len(strategies_to_use):
                try:
                    frequency_values = [float(val.strip()) for val in values]
                    total_frequency = sum(frequency_values)
                    if total_frequency == 0:
                        messagebox.showerror("Error", "Total frequency cannot be zero.")
                        return
                    frequency_values = [val / total_frequency for val in frequency_values]
                    output_text.delete(1.0, tk.END)
                    output_text.insert(tk.END, f"Manual: {', '.join([f'{strategy} = {freq:.3f}' for strategy, freq in zip(strategies_to_use, frequency_values)])}")
                    strategy_frequency.append((selected_frequency_option, frequency_values))

                    for strategy, freq in zip(strategies_to_use, frequency_values):
                        strategy_freq_dict[strategy] = freq

                except ValueError:
                    messagebox.showerror("Error", "Invalid frequency values. Please enter numeric values.")
            else:
                messagebox.showerror("Error", f"Number of frequency values should be equal to the number of strategies ({len(strategies_to_use)}).")
        else:
            messagebox.showerror("Error", "Invalid strategy frequency option.")
    else:
        messagebox.showerror("Error", "Please enter valid population size and select strategies.")

# Global variables to store the data
disruption_times_data = []
matrices_data = []

def open_payoff_matrix_change_window():
    def create_matrices():
        global saved_disruption_times, saved_matrices, disruption_times_data, matrices_data  # Access the global lists to update data
        try:
            num_matrices = int(entry_num_matrices.get())
            selected_strategies = selected_strategies_listbox.curselection()


            if num_matrices < 1:
                messagebox.showerror("Error", "Number of matrices should be a positive integer.")
                return

            disruption_times = entry_disruption_times.get().split(',')

            if len(disruption_times) != num_matrices:
                messagebox.showerror("Error", "The number of disruption times should match the number of matrices.")
                return
            
            disruption_times = [int(time.strip()) for time in entry_disruption_times.get().split(',')]
            matrices = []

            matrix_values = entry_matrix_values.get().split(';')
            if len(matrix_values) != num_matrices:
                messagebox.showerror("Error", f"Please enter exactly {num_matrices} matrices separated by ';'.")
                return

            for idx, matrix_str in enumerate(matrix_values):
                matrix_elements = matrix_str.split(',')
                matrix_size = int(len(matrix_elements)**0.5)
                # Ensure the matrix size does not exceed the number of original strategies
                if matrix_size > len(original_strategy_names):
                    messagebox.showerror("Error", "Matrix size exceeds number of strategies.")
                    return
                if len(matrix_elements) != matrix_size * matrix_size:
                    messagebox.showerror("Error", f"Please enter exactly {matrix_size * matrix_size} elements for each {matrix_size}x{matrix_size} matrix.")
                    return
                matrix = [list(map(float, matrix_elements[j:j + matrix_size])) for j in range(0, len(matrix_elements), matrix_size)]
                matrices.append((matrix, f"t{idx + 1} = {disruption_times[idx]}"))


            saved_disruption_times = disruption_times
            saved_matrices = matrices

            display_output.config(state=tk.NORMAL)
            display_output.delete("1.0", tk.END)

            output = "\nGenerated Matrices:\n"
            for idx, (matrix, time_str) in enumerate(saved_matrices):
                output += f"\nMatrix {idx+1}: {time_str}\n"
                for row in matrix:
                    output += str(row) + "\n"

            display_output.insert(tk.END, output)
            display_output.config(state=tk.DISABLED)

        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter valid integers for the number of matrices and disruption time.")

    def clear_display():
        entry_num_matrices.delete(0, tk.END)
        entry_disruption_times.delete(0, tk.END)
        entry_matrix_values.delete(0, tk.END)
        display_output.config(state=tk.NORMAL)
        display_output.delete("1.0", tk.END)
        display_output.config(state=tk.DISABLED)

    def save_data():
        global saved_disruption_times, saved_matrices, disruption_times_data, matrices_data  # Access the global lists to save data
        if saved_matrices:
            # Store the data in corresponding variables
            matrices_data = [matrix for matrix, _ in saved_matrices]
            disruption_times_data = saved_disruption_times
            messagebox.showinfo("Saved", "Data has been saved.")
        else:
            messagebox.showwarning("Nothing to Save", "No data to save.")

    payoff_window = tk.Toplevel(root)
    payoff_window.title("Pay-off Matrix Change")

    # Input widgets for Pay-off Matrix Change window
    label_num_matrices = tk.Label(payoff_window, text="Number of matrices:")
    entry_num_matrices = tk.Entry(payoff_window)
    label_disruption_times = tk.Label(payoff_window, text="Disruption Time (comma-separated):")
    entry_disruption_times = tk.Entry(payoff_window)
    label_matrix_values = tk.Label(payoff_window, text="Enter matrix values (nxn separated by ';', elements by ','):")
    entry_matrix_values = tk.Entry(payoff_window)
    btn_generate = tk.Button(payoff_window, text="Generate Matrices", command=create_matrices)
    btn_clear = tk.Button(payoff_window, text="Clear", command=clear_display)
    btn_save_data = tk.Button(payoff_window, text="Save", command=save_data)

    # Output widget for Pay-off Matrix Change window
    display_output = tk.Text(payoff_window, height=10, width=60, state=tk.DISABLED)

    # Layout for Pay-off Matrix Change window
    label_num_matrices.grid(row=0, column=0, sticky="w")
    entry_num_matrices.grid(row=0, column=1, padx=5, pady=5)
    label_disruption_times.grid(row=1, column=0, sticky="w")
    entry_disruption_times.grid(row=1, column=1, padx=5, pady=5)
    label_matrix_values.grid(row=2, column=0, columnspan=2, sticky="w")
    entry_matrix_values.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
    btn_generate.grid(row=4, column=0, padx=5, pady=5)
    btn_clear.grid(row=4, column=1, padx=5, pady=5)
    btn_save_data.grid(row=6, column=0, columnspan=2, padx=5, pady=5)
    display_output.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

def reset_payoff_matrix_change_window():
    global disruption_times_data, matrices_data, disruption_type_payoff  # Access the global lists and the checkbox variable
    disruption_times_data = []
    matrices_data = []
    disruption_type_payoff.set(False)  # Uncheck the "Pay-off matrix change" option
    
new_matrices = []
disruption_new_strategy_times = []
New_Strategies_list = []


def open_new_strategy_window():
    global New_Strategies_list
    global strategy_freq_dict
    selected_new_strategies = []
    disruption_times_entry = None

    def add_new_strategy():
        nonlocal selected_new_strategies

        selected_indices = selected_new_strategies_entry.get().strip()
        if not selected_indices:
            messagebox.showerror("Error", "Please enter at least one strategy name.")
            return

        new_strategies = [strategy.strip() for strategy in selected_indices.split(',')]

        # Store the selected new strategies in the order they were selected
        selected_new_strategies = new_strategies

        # Add the new strategies to the New_Strategies_list
        New_Strategies_list.extend(new_strategies)

    def save_new_strategy_data():
        nonlocal disruption_times_entry
        global filtered_matrix, filtered_strategy_names, new_matrices, disruption_new_strategy_times

        if not selected_new_strategies:
            messagebox.showerror("Error", "Please enter new strategies before saving.")
            return

        disruption_times = disruption_times_entry.get().split(',')
        
        if len(selected_new_strategies) != len(disruption_times):
            messagebox.showerror("Error", "The number of disruption times should match the number of new strategies entered.")
            return

        try:
            disruption_times = [int(val.strip()) for val in disruption_times]
        except ValueError:
            messagebox.showerror("Error", "Invalid disruption times. Please enter numeric values separated by commas.")
            return

        # Check if any new strategy name already exists in the filtered strategy names
        existing_strategy_names = set(filtered_strategy_names)
        new_strategy_names = set(selected_new_strategies)
        if any(name in existing_strategy_names for name in new_strategy_names):
            conflicting_names = new_strategy_names.intersection(existing_strategy_names)
            messagebox.showerror("Error", f"Mentioned new strategy(s) already exist in the pay-off matrix: {', '.join(conflicting_names)}.")
            return

        # Clear the list of new matrices and disruption times
        new_matrices.clear()
        disruption_new_strategy_times.clear()

        new_filtered_strategy_names = filtered_strategy_names.copy()

        for new_strategy in selected_new_strategies:
            # Add the new strategy to the filtered names list if it's not already present
            if new_strategy not in new_filtered_strategy_names:
                new_filtered_strategy_names.append(new_strategy)

            # Create a new matrix for each selected new strategy
            filtered_matrix_new_strategy = []

            for row_strategy in new_filtered_strategy_names:
                strategy_index = original_strategy_names.index(row_strategy)
                new_row = [original_matrix[strategy_index][original_strategy_names.index(col)] for col in new_filtered_strategy_names]
                filtered_matrix_new_strategy.append(new_row)

            # Store the new matrix and disruption time
            new_matrices.append(filtered_matrix_new_strategy)
            disruption_new_strategy_times.append(disruption_times[selected_new_strategies.index(new_strategy)])

        # Update the output_text in the main window to display the new matrices and disruption times
        check_variable_contents()

        # Clear the entry widgets
        selected_new_strategies_entry.delete(0, tk.END)
        disruption_times_entry.delete(0, tk.END)

        # Close the "New Strategy" window
        new_strategy_window.destroy()

    
    new_strategy_window = tk.Toplevel(root)
    new_strategy_window.title("New Strategy")

    # Create an entry widget for the user to enter the new strategy names
    selected_new_strategies_label = tk.Label(new_strategy_window, text="Enter New Strategy Names (comma-separated):")
    selected_new_strategies_label.pack(pady=5)
    selected_new_strategies_entry = tk.Entry(new_strategy_window)
    selected_new_strategies_entry.pack(pady=5)

    # Create an entry widget for the user to enter the disruption times
    disruption_times_label = tk.Label(new_strategy_window, text="Disruption Times (comma-separated):")
    disruption_times_label.pack(pady=5)
    disruption_times_entry = tk.Entry(new_strategy_window)
    disruption_times_entry.pack(pady=5)

    # Create a button to save the selected new strategy
    save_button = tk.Button(new_strategy_window, text="Save Strategy", command=add_new_strategy)
    save_button.pack(pady=5)

    # Create a button to save the new strategy data
    save_data_button = tk.Button(new_strategy_window, text="Save Data", command=save_new_strategy_data)
    save_data_button.pack(pady=5)

def reset_new_strategy_window():
    global new_matrices, disruption_new_strategy_times, disruption_type_new_strategy,New_Strategies_list  # Access the global lists and the checkbox variable
    new_matrices = []
    disruption_new_strategy_times = []
    New_Strategies_list = []
    disruption_type_new_strategy.set(False)  # Uncheck the "New Strategy" option
   

# Add these two lists to store the data for Strategy Popularity window
Strategy_popularity_strategies = []
strat_pop_new_freq = [] 
Strat_pop_drisruption_time = [] 

strategy_popularity_checked = None
strategy_names_entry_popup = None
strategy_popularity_values_entry = None
disruption_time_entry = None

def open_strategy_popularity_window():
    global strategy_popularity_checked, strategy_names_entry_popup, strategy_popularity_values_entry
  
    strategy_popularity_window = tk.Toplevel(root)
    strategy_popularity_window.title("Strategy Popularity")

    def validate_strategy_names():
        # Validate the entered strategy names against the strategy names in the main window
        main_window_strategy_names = strategy_names_entry.get().split(',')
        main_window_strategy_names = [name.strip() for name in main_window_strategy_names]
        entered_strategy_names = strategy_names_entry_popup.get().split(',')
        entered_strategy_names = [name.strip() for name in entered_strategy_names]

        for name in entered_strategy_names:
            if name not in main_window_strategy_names:
                messagebox.showerror("Error", "Only strategies in the pay-off matrix should be entered.")
                return False
        return True

    def validate_popularity_values():
        popularity_values = [val.strip() for val in strategy_popularity_values_entry.get().split(',')]
        try:
            # Convert the disruption time values to floats and check if they are <= 1.0
            disruption_times = [float(val.strip()) for val in popularity_values]
            for time in disruption_times:
                if time > 1.0:
                    messagebox.showerror("Error", "Popularity values should be less than or equal to 1.0.")
                    return False
        except ValueError:
            messagebox.showerror("Error", "Invalid popularity value. Please enter valid decimal values separated by commas.")
            return False
        return True

    def save_strategy_popularity_data():
        if not validate_strategy_names():
            return

        if not validate_popularity_values():
            return

        strategies = strategy_names_entry_popup.get()
        popularity_values = strategy_popularity_values_entry.get()

        Strategy_popularity_strategies.clear()
        strat_pop_new_freq .clear()
        Strat_pop_drisruption_time.clear()

        strategies_list = [name.strip() for name in strategies.split(',')]
        popularity_list = [val.strip() for val in popularity_values.split(',')]
        Strategy_popularity_strategies.extend(strategies_list)

        if len(strategies_list) != len(popularity_list):
            messagebox.showerror("Error", "The number of strategies should match the number of popularity values.")
            return

        # Convert the disruption time values to floats before storing them
        disruption_freq = [float(val.strip()) for val in popularity_values.split(',')]
        strat_pop_new_freq .extend(disruption_freq)

        # Get the disruption time values from the entry box
        disruption_time_values = disruption_time_entry.get()
        disruption_time_values = [int(val.strip()) for val in disruption_time_values.split(',')]
        Strat_pop_drisruption_time.extend(disruption_time_values)  # Store in the disruption time list

        messagebox.showinfo("Saved", "Data has been saved.")
        strategy_popularity_window.destroy()

    strategy_names_label_popup = tk.Label(strategy_popularity_window, text="Strategies (comma-separated):")
    strategy_names_label_popup.pack(pady=5)

    strategy_names_entry_popup = tk.Entry(strategy_popularity_window)
    strategy_names_entry_popup.pack(pady=5)

    strategy_popularity_values_label = tk.Label(strategy_popularity_window, text="Popularity (Strategy Frequency) (comma-separated decimals <= 1.0):")
    strategy_popularity_values_label.pack(pady=5)

    strategy_popularity_values_entry = tk.Entry(strategy_popularity_window)
    strategy_popularity_values_entry.pack(pady=5)

    # Create the disruption time entry widget
    disruption_time_label = tk.Label(strategy_popularity_window, text="Disruption Time (comma-separated decimals):")
    disruption_time_label.pack(pady=5)
    
    disruption_time_entry = tk.Entry(strategy_popularity_window)
    disruption_time_entry.pack(pady=5)

    save_button = tk.Button(strategy_popularity_window, text="Save", command=save_strategy_popularity_data)
    save_button.pack(pady=5)

def reset_strategy_popularity():
    global Strategy_popularity_strategies,strat_pop_new_freq, strategy_popularity_checked, strategy_names_entry_popup, strategy_popularity_values_entry
    Strategy_popularity_strategies = []
    strat_pop_new_freq = [] 
    disruption_type_strategy_popularity.set(False)  # Uncheck the "Strategy Popularity" option
    

add_pop = []
add_pop_time = []

def open_player_population_window():
    # Create a new Toplevel window
    player_population_window = tk.Toplevel(root)
    player_population_window.title("Player Population")
    
    # Create labels and entry widgets
    add_pop_label = tk.Label(player_population_window, text="New Population:")
    add_pop_label.grid(row=0, column=0, padx=10, pady=10)
    add_pop_entry = tk.Entry(player_population_window)
    add_pop_entry.grid(row=0, column=1, padx=10, pady=10)
    
    add_pop_time_label = tk.Label(player_population_window, text=" Disruption Time:")
    add_pop_time_label.grid(row=1, column=0, padx=10, pady=10)
    add_pop_time_entry = tk.Entry(player_population_window)
    add_pop_time_entry.grid(row=1, column=1, padx=10, pady=10)



    # Function to save the entered values
    def save_values():
        # Fetch values from entries
        add_pop_values = [int(x) for x in add_pop_entry.get().split(',')]if add_pop_entry.get() else []
        add_pop_time_values = [int(x) for x in add_pop_time_entry.get().split(',')]if add_pop_time_entry.get() else []
        
        # Check if counts match
        if len(add_pop_values) != len(add_pop_time_values):
            messagebox.showerror("Error", "Count mismatch between Add Population and Add Population Time.")
            return
        
        
        # Check time values against "Time Steps" from the main window
        time_steps = get_time_steps()  # Assuming this function fetches the time steps value
        invalid_times = [x for x in add_pop_time_values if x > time_steps]
        if invalid_times:
            messagebox.showerror("Error", f"Invalid times: {', '.join(map(str, invalid_times))}. Exceeds Time Steps value.")
            return

        # Save to lists
        add_pop.extend(add_pop_values)
        add_pop_time.extend(add_pop_time_values)

        # Close the window after saving
        player_population_window.destroy()

    save_button = tk.Button(player_population_window, text="Save", command=save_values)
    save_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

def reset_player_population():
    # Reset the Entry fields in the "Player Population" window to empty
    add_pop.clear()
    add_pop_time.clear()
    # Uncheck the tickbox related to "Player Population"
    disruption_type_player_population.set(False)  # Assuming player_population_var is the BooleanVar for the tickbox

pop_size = None
time_step = None

def get_population_size():
    global pop_size
    population_size = population_size_entry.get()
    if population_size:
        try:
            pop_size = float(population_size)
            return pop_size
        except ValueError:
            messagebox.showerror("Error", "Population size should be a numeric value.")
            return None
    pop_size = 100
    return pop_size

def get_time_steps():
    global time_step
    time_steps = time_steps_entry.get()
    if time_steps:
        try:
            time_step = int(time_steps)
            return time_step
        except ValueError:
            messagebox.showerror("Error", "Time steps should be an integer.")
            return None
    time_step = 100
    return time_step

import sys
from tkinter import filedialog

original_stdout = None  # Declare a global variable to store the original standard output

def start_capture():
    global original_stdout
    original_stdout = sys.stdout
    sys.stdout = open('output.txt', 'w')

def stop_capture():
    global original_stdout
    sys.stdout.close()
    sys.stdout = original_stdout

def download_output():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="output.txt")
    if not file_path:
        return
    with open('output.txt', 'r') as f:
        content = f.read()
    with open(file_path, 'w') as f:
        f.write(content)

import pandas as pd
from tkinter import ttk

def display_with_output_button(results,global_match_record):
    graph_window = tk.Toplevel(root)
    graph_window.geometry("1200x700")
    graph_window.title("Graphical Representation")

    
    def show_results_in_graph_window():
        # Clear the graph_window content
        for widget in graph_window.winfo_children():
            widget.destroy()

        # Create frames for the Pac-Man animation and results
        pacman_frame = tk.Frame(graph_window, width = 600)
        pacman_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        results_frame = tk.Frame(graph_window, width = 400)
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a canvas in the Pac-Man frame and display the animation
        canvas = tk.Canvas(pacman_frame, width=600, height=400)
        canvas.pack(pady=50) 
        pacman_dominance(results, canvas, pacman_frame)

        # Create DataFrame from results and filter columns
        df = pd.DataFrame(results)
        df = df[['time_step', 'winners', 'losers', 'disruption']]
        
        # Set style for Treeview to have the desired background color
        style = ttk.Style(graph_window)
        style.configure("Custom.Treeview", background="#f0efeb", bordercolor="#B85C4", 
                        foreground="Black", fieldbackground="#D8E2DC")
        style.configure("Custom.Treeview.Heading", background="#B85C4",bordercolor="black", 
                        foreground="black")

        # Create a Treeview with custom style
        tree = ttk.Treeview(results_frame, style="Custom.Treeview", columns=list(df.columns), show='headings')
        for column in df.columns:
            tree.heading(column, text=column)
            tree.column(column, width=100)

        tree.pack(fill=tk.BOTH, expand=True)
        for index, row in df.iterrows():
            tree.insert("", "end", values=list(row))

        # Add a "Back" button
        back_button = tk.Button(pacman_frame, text="Back", command=recreate_graph_content)
        back_button.pack(pady=10, side=tk.BOTTOM)

        # Add a "Download Output" button
        download_output_button = tk.Button(graph_window, text="Download Output", command=download_output)
        download_output_button.pack(pady=10, side=tk.BOTTOM)


    def show_strategy_plotter_window():
        print("Inside the show_strategy_plotter_window function")

        scatter_plot_window = tk.Toplevel(graph_window)
        scatter_plot_window.geometry("1200x600")
        scatter_plot_window.title("Strategy Plotter")

        show_strategy_plotter(scatter_plot_window, results, global_match_record)


    def recreate_graph_content():
        # Recreate the graphical content in the graph_window
        for widget in graph_window.winfo_children():
            widget.destroy()

        display_graphical_representation_dynamic_v2(results, graph_window)

        output_button = tk.Button(graph_window, text="Output", command=show_results_in_graph_window)
        output_button.pack(pady=60, side=tk.BOTTOM)

        # Add a "Show Strategy Plotter" button
        show_strategy_plotter_button = tk.Button(graph_window, text="Show Strategy Plotter", command=lambda: show_strategy_plotter_window())
        show_strategy_plotter_button.pack(pady=10, side=tk.BOTTOM)

    recreate_graph_content()


def handle_submit():
    global original_strategy_names, original_payoff_matrix

    if filtered_matrix and filtered_strategy_names:
        Initial_matrix = filtered_matrix
        Initial_strategy_names = filtered_strategy_names
    else:
        Initial_matrix = original_payoff_matrix
        Initial_strategy_names = original_strategy_names

    strategy_frequencies = strategy_freq_dict
    population_size = get_population_size()
    time_steps = get_time_steps()

    disruption_time_payoff_matrix_change_list = disruption_times_data
    Pay_off_matrix_change_list = matrices_data
    disruption_matrices_new_strategies = new_matrices
    disruption_time_new_strategy_list = disruption_new_strategy_times
    new_strategy_name_list = New_Strategies_list
    Strategy_Names_for_pop = Strategy_popularity_strategies
    New_strategy_freq_pop = strat_pop_new_freq
    distruption_time_strat_pop = Strat_pop_drisruption_time
    disrup_pop_time_list = add_pop_time
    disrup_pop_value = add_pop

    # Run the simulation
    start_capture()
    matrix = convert_matrix_to_dict(Initial_strategy_names, Initial_matrix)
    results, global_match_record  = simulate_Metasimul8(
        strategy_frequencies, population_size, time_steps, matrix,
        disruption_time_payoff_matrix_change_list, Pay_off_matrix_change_list,
        new_strategy_name_list, disruption_matrices_new_strategies,
        disruption_time_new_strategy_list, disrup_pop_time_list,
        disrup_pop_value, Strategy_Names_for_pop, New_strategy_freq_pop,
        distruption_time_strat_pop
    )

    stop_capture()

    # Display the graphical representation with the output button using the wrapper function
    display_with_output_button(results, global_match_record)


def check_variable_contents():
    global strategy_freq_dict, filtered_matrix, filtered_strategy_names, strategy_frequency, disruption_times_data, matrices_data, new_matrices, disruption_new_strategy_times

    output_text.delete(1.0, tk.END)

    output_text.insert(tk.END, "Main Window Variables:\n")
    output_text.insert(tk.END, f"Number of Strategies: {num_strategies_entry.get()}\n")
    output_text.insert(tk.END, "\nOriginal Payoff Matrix:\n")
    output_text.insert(tk.END, f"{original_payoff_matrix}\n")
    output_text.insert(tk.END, "\nFiltered Matrix:\n")
    output_text.insert(tk.END, f"{filtered_matrix}\n")
    output_text.insert(tk.END, "\nOriginal Strategy Names:\n")
    output_text.insert(tk.END, f"{original_strategy_names}\n")
    output_text.insert(tk.END, "\nFiltered Strategy Names:\n")
    output_text.insert(tk.END, f"{filtered_strategy_names}\n")
    output_text.insert(tk.END, "\nStrategy Frequency:\n")
    output_text.insert(tk.END, f"{strategy_freq_dict}\n")

    if disruption_times_data and matrices_data:
        output_text.insert(tk.END, "\nDisruption Times:\n")
        output_text.insert(tk.END, f"{disruption_times_data}\n")
        output_text.insert(tk.END, "\nDisruption Payoffs:\n")
        output_text.insert(tk.END, f"{matrices_data}\n")
    else:
        output_text.insert(tk.END, "\nPay-off Matrix Change Variables: No data found.\n")

    # Display the new matrices and disruption times for new strategies
    if New_Strategies_list and new_matrices and disruption_new_strategy_times:
        output_text.insert(tk.END, "\nNew Matrices for New Strategies:\n")
        for idx, matrix in enumerate(new_matrices):
            output_text.insert(tk.END, f"\nMatrix {idx + 1}:\n")
            for row in matrix:
                output_text.insert(tk.END, str(row) + "\n")

        output_text.insert(tk.END, "\nDisruption matrices for New Strategies:\n")
        output_text.insert(tk.END, f"{new_matrices}\n")

        output_text.insert(tk.END, "\nDisruption Times for New Strategies:\n")
        output_text.insert(tk.END, f"{disruption_new_strategy_times}\n")

        output_text.insert(tk.END, "\n New Strategies Names:\n")
        output_text.insert(tk.END, f"{New_Strategies_list}\n")
    else:
        output_text.insert(tk.END, "\nNew Strategy Variables: No data found.\n")

    if Strategy_popularity_strategies and strat_pop_new_freq and Strat_pop_drisruption_time:
        output_text.insert(tk.END, "\nStrategy Popularity Variables:\n")
        output_text.insert(tk.END, f"Strategies: {', '.join(Strategy_popularity_strategies)}\n")
        output_text.insert(tk.END, f"Popularity (Strategy Frequency): {', '.join(map(str, strat_pop_new_freq))}\n")
        output_text.insert(tk.END, f"Disruption time: {', '.join(map(str, Strat_pop_drisruption_time))}\n")
    else:
        output_text.insert(tk.END, "\nStrategy Popularity Variables: No data found.\n")
    
    if add_pop and add_pop_time:
        output_text.insert(tk.END, "\n New population:\n")
        output_text.insert(tk.END, f"{add_pop}\n")
        output_text.insert(tk.END, "\nDisruption time:\n")
        output_text.insert(tk.END, f"{add_pop_time}\n")
    else:
        output_text.insert(tk.END, "\nAdd population and Add population time: No data found\n")


root = tk.Tk()
root.title("MetaSimul8")
root.geometry("900x700") 

root.grid_rowconfigure(0, weight=1)  # main row
root.grid_columnconfigure(0, weight=3)  # left column (e.g., for plot_frame)
root.grid_columnconfigure(1, weight=1)  # right column (e.g., for control_frame)


# Load the image
original_image = Image.open("log2_crop.png")
    
# Resize the image
resized_image = original_image.resize((600, 500), Image.ANTIALIAS)
    
# Convert the resized image for use in tkinter
bg_image = ImageTk.PhotoImage(resized_image)

# Load the background image
#bg_image = PhotoImage(file="log2_crop.png")

# Create a label with the loaded image
bg_label = Label(root, image=bg_image)
bg_label.place(relwidth=1, relheight=1)

submit_button = tk.Button(root, text="Submit", command=handle_submit)
submit_button.grid (row=18, column=2, padx=5, pady=5)

reset_player_population_button = tk.Button(root, text="Reset - Player Population", command=reset_player_population)
reset_player_population_button.grid(row=15, column=3, padx=5, pady=5)

reset_strategy_popularity_button = tk.Button(root, text="Reset - Strategy Popularity", command=reset_strategy_popularity)
reset_strategy_popularity_button.grid(row=15, column=2, padx=5, pady=5)

# Create the "Reset - New Strategy" button
reset_new_strategy_button = tk.Button(root, text="Reset - New Strategy", command=reset_new_strategy_window)
reset_new_strategy_button.grid(row=15, column=1, padx=5, pady=5)

reset_payoff_button = tk.Button(root, text="Reset - Pay Off Matrix Change", command=reset_payoff_matrix_change_window)
reset_payoff_button.grid(row=15, column=0, padx=5, pady=5)

# Add a button to the GUI to check the contents
check_contents_button = tk.Button(root, text="Check", command=check_variable_contents)
check_contents_button.grid(row=18, column=1, padx=5, pady=5)

num_strategies_label = tk.Label(root, text="No. of Strategies:")
num_strategies_label.grid(row=0, column=0, padx=5, pady=5)

num_strategies_entry = tk.Entry(root)
num_strategies_entry.grid(row=0, column=1, padx=5, pady=5)

strategy_names_label = tk.Label(root, text="Strategy Names (comma-separated):")
strategy_names_label.grid(row=1, column=0, padx=5, pady=5)

strategy_names_entry = tk.Entry(root)
strategy_names_entry.grid(row=1, column=1, padx=5, pady=5)

matrix_values_label = tk.Label(root, text="Values:")
matrix_values_label.grid(row=2, column=0, padx=5, pady=5)

matrix_values_entry = tk.Text(root, width=40, height=6)
matrix_values_entry.grid(row=2, column=1, padx=5, pady=5)

population_size_label = tk.Label(root, text="Population Size (Optional, default=100):")
population_size_label.grid(row=3, column=0, padx=5, pady=5)

population_size_entry = tk.Entry(root)
population_size_entry.grid(row=3, column=1, padx=5, pady=5)
population_size_entry.insert(tk.END, "100")

time_steps_label = tk.Label(root, text="Time Steps (Optional, default=100):")
time_steps_label.grid(row=4, column=0, padx=5, pady=5)

time_steps_entry = tk.Entry(root)
time_steps_entry.grid(row=4, column=1, padx=5, pady=5)
time_steps_entry.insert(tk.END, "100")

filter_var = tk.BooleanVar()
filter_check = tk.Checkbutton(root, text="Filter", variable=filter_var, command=filter_data_gui)
filter_check.grid(row=5, column=0, padx=5, pady=5)

# Create a frame to hold the listbox and scrollbar
listbox_frame = tk.Frame(root)
listbox_frame.grid(row=5, column=1, padx=5, pady=5)

selected_strategies_var = tk.StringVar()
selected_strategies_listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE, listvariable=selected_strategies_var, width=20, height=4)
selected_strategies_listbox.pack(side=tk.LEFT, fill=tk.BOTH)

# Add scrollbar to the listbox
listbox_scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=selected_strategies_listbox.yview)
listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the listbox to work with the scrollbar
selected_strategies_listbox.config(yscrollcommand=listbox_scrollbar.set)

strategy_frequency_label = tk.Label(root, text="Strategy Frequency:")
strategy_frequency_label.grid(row=6, column=0, padx=5, pady=5)

strategy_frequency_var = tk.StringVar()
strategy_frequency_options = ["Equal Division", "Manual"]
strategy_frequency_var.set(strategy_frequency_options[0])

strategy_frequency_menu = tk.OptionMenu(root, strategy_frequency_var, *strategy_frequency_options, command=update_strategy_frequency_option)
strategy_frequency_menu.grid(row=6, column=1, padx=5, pady=5)

value_label = tk.Label(root, text="Value (if Manual):")
value_entry = tk.Entry(root)
value_label.grid(row=7, column=0, padx=4, pady=4)

disruption_type_label = tk.Label(root, text="Disruption Type")
disruption_type_label.grid(row=13, column=1, padx=5, pady=5)

# Integrate the Pay-off Matrix Change window
disruption_type_payoff = tk.BooleanVar()
disruption_type_payoff_checkbox = tk.Checkbutton(root, text="Pay-off matrix change", variable=disruption_type_payoff, command=open_payoff_matrix_change_window)
disruption_type_payoff_checkbox.grid(row=14, column=0, padx=5, pady=5)

disruption_type_new_strategy = tk.BooleanVar()
disruption_type_new_strategy_checkbox = tk.Checkbutton(root, text="New Strategy", variable=disruption_type_new_strategy, command=open_new_strategy_window)
disruption_type_new_strategy_checkbox.grid(row=14, column=1, padx=5, pady=5)

disruption_type_strategy_popularity = tk.BooleanVar()
disruption_type_strategy_popularity_checkbox = tk.Checkbutton(root, text="Strategy Popularity", variable=disruption_type_strategy_popularity,command=open_strategy_popularity_window)
disruption_type_strategy_popularity_checkbox.grid(row=14, column=2, padx=5, pady=5)

disruption_type_player_population = tk.BooleanVar()
disruption_type_player_population_checkbox = tk.Checkbutton(root, text="Player Population", variable=disruption_type_player_population, command=open_player_population_window)
disruption_type_player_population_checkbox.grid(row=14, column=3, padx=5, pady=5)

create_matrix_button = tk.Button(root, text="Create Matrix", command=create_matrix_table)
create_matrix_button.grid(row=2, column=2, columnspan=2, padx=5, pady=5)

apply_filter_button = tk.Button(root, text="Apply Filter", command=apply_filter)
apply_filter_button.grid(row=5, column=2, columnspan=2, padx=5, pady=5)

calculate_frequency_button = tk.Button(root, text="Calculate Strategy Frequency", command=calculate_strategy_frequency)
calculate_frequency_button.grid(row=6, column=2, columnspan=2, padx=5, pady=5)

output_text = tk.Text(root, height=10, width=60)
output_text.grid(row=16, column=0, columnspan=4, padx=5, pady=5)

root.protocol("WM_DELETE_WINDOW", on_exit)
root.mainloop()
