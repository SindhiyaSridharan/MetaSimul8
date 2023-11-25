
# This is the main backend simulation code

import random
import copy

# Nested Dictionary : Matrix Conversion 

def convert_matrix_to_dict(strategy_names, matrix_values):
    return {strategy: {other_strategy: matrix_values[i][j] 
                       for j, other_strategy in enumerate(strategy_names)} 
            for i, strategy in enumerate(strategy_names)}

# Quantification Unit

def best_response_against(opponent_strategy, matrix):
    # Calculate the difference in payoffs for each strategy against the opponent strategy
    payoff_differences = {strategy: matrix[strategy][opponent_strategy] - matrix[opponent_strategy][strategy] 
                          for strategy in matrix}
    
    # Get the strategy with the maximum payoff difference against the opponent strategy
    best_response = max(payoff_differences, key=payoff_differences.get)
    
    return best_response


def calculate_new_frequencies(all_strategies):
    strategy_freqs = {}
    total_count = len(all_strategies)
    
    for strategy in set(all_strategies): 
        strategy_freqs[strategy] = all_strategies.count(strategy) / total_count

    return strategy_freqs


def calculate_expected_payoff(current_frequencies, matrix):
    expected_payoffs = {}
    
    # Created an updated frequency dictionary that includes all strategies
    updated_current_frequency = {strategy: current_frequencies.get(strategy, 0) for strategy in matrix.keys()}
    
    for strategy in matrix.keys():  # Loop through all strategies in the payoff matrix
        expected_payoff = 0
        for opponent in updated_current_frequency:
            expected_payoff += updated_current_frequency[opponent] * matrix[strategy][opponent]
        expected_payoffs[strategy] = expected_payoff

    print("Expected Payoffs:", expected_payoffs)
    
    return expected_payoffs


def calculate_win_percentage(matches):
    win_counters = {}
    
    # Initialize counters for each matchup
    for match in matches:
        winner, loser = match
        key = f"{winner}_vs_{loser}"
        
        # If the key isn't in the dictionary, initialize it
        if key not in win_counters:
            win_counters[key] = 0
        
        win_counters[key] += 1

    # Calculate percentages
    total_matches = len(matches)
    win_percentages = {matchup: (count / total_matches) * 100 for matchup, count in win_counters.items()}
    
    return win_percentages


def calculate_overall_win_percentages(global_match_record):


    total_matches = sum(global_match_record.values())
    win_percentages = {}

    for (winner, loser), count in global_match_record.items():
        percentage = (count / total_matches) * 100
        win_percentages[f"{winner} vs {loser}"] = f"{winner} wins {percentage:.2f}% of the time"

    return win_percentages


# Disruption Unit

def update_payoff_matrix(current_step, current_matrix, disruption_times=None, new_matrices=None):
    
    if disruption_times and current_step in disruption_times:
        idx = disruption_times.index(current_step)
        new_matrix_values = new_matrices[idx]
        strategies = list(current_matrix.keys())
        new_matrix = {}
        for i, s1 in enumerate(strategies):
            new_matrix[s1] = {}
            for j, s2 in enumerate(strategies):
                new_matrix[s1][s2] = new_matrix_values[i][j]
        
        print(f"\nPayoff Matrix Updated at Time Step {current_step}!\nNew Matrix:\n{new_matrix}\n")
        return new_matrix
    return current_matrix



def add_new_strategy(strategy_frequencies, current_matrix, new_strategy_name, matrix_values):

    total_existing_frequency = 1
    number_of_existing_strategies = len(strategy_frequencies)

    # Copy the existing frequencies as they will remain the same.
    adjusted_frequencies = strategy_frequencies.copy()

    # Add the new strategy with frequency 0
    adjusted_frequencies[new_strategy_name] = 0
    
    # Merge the strategy names
    all_strategy_names = list(current_matrix.keys()) + [new_strategy_name]
    
    # Convert the matrix values into a dictionary format
    updated_matrix = convert_matrix_to_dict(all_strategy_names, matrix_values)
    
    # Update the current matrix with the new strategy's payoffs
    for strategy, payoffs in updated_matrix.items():
        current_matrix[strategy] = payoffs

    return adjusted_frequencies, current_matrix

def adjust_remaining_frequencies(current_frequencies, strategy_to_adjust, new_frequency):
    # Directly set the strategy's frequency to the new value
    current_frequencies[strategy_to_adjust] = round(new_frequency, 1)
    
    # Calculate the total frequency that needs to be distributed among other strategies
    total_remaining_freq = 1 - current_frequencies[strategy_to_adjust]

    # Remove the strategy_to_adjust to get the list of other strategies
    other_strategies = {k: v for k, v in current_frequencies.items() if k != strategy_to_adjust}
    
    # Find the sum of other strategies' frequencies
    sum_other_freq = sum(other_strategies.values())

    # Distribute the remaining frequency among other strategies proportionally
    for strategy, freq in other_strategies.items():
        proportion = freq / sum_other_freq
        current_frequencies[strategy] = round(total_remaining_freq * proportion, 1)
    
    # Correct any discrepancies from rounding so that frequencies sum up to 1
    error = 1.0 - sum(current_frequencies.values())
    if error != 0.0:
        # Find a strategy with non-zero frequency to adjust
        for strat, freq in current_frequencies.items():
            if freq > 0:
                current_frequencies[strat] += error
                break
    
    return current_frequencies


def Strategy_Popularity(time_step, Strategy_Names_for_pop, New_strategy_freq_pop, distruption_time_strat_pop, current_frequencies, population_size):
 
    if (time_step+1) not in distruption_time_strat_pop:
        return current_frequencies
    
    
    index = distruption_time_strat_pop.index(time_step+1)

    if index >= len(Strategy_Names_for_pop) or index < 0:
        print(f"Index {index} out of range for list of length {len(Strategy_Names_for_pop)}!")
        return current_frequencies
    

    strategy_to_adjust = Strategy_Names_for_pop[index]
    new_frequency = New_strategy_freq_pop[index]  # Renaming desired_frequency to new_frequency
    
    # Calculate new number for desired strategy
    desired_num = new_frequency * population_size
    
    # Rest of the population after assigning desired numbers to the strategy
    rest_population = population_size - desired_num
    
    # Calculate the new number of individuals for each strategy based on the old frequencies
    new_numbers = {}
    for strat, freq in current_frequencies.items():
        new_numbers[strat] = freq * rest_population
    
    # Calculate difference between rest_population and sum of new_numbers
    diff_population = rest_population - sum(new_numbers.values())
    
    # List of strategies that existed before (excluding the strategy we want to adjust)
    existed_strategies = [s for s in current_frequencies.keys() if current_frequencies[s] != 0]
    
    # Distribute the diff_population equally among the existing strategies
    adjustment_per_strategy = diff_population / len(existed_strategies)
    
    for strat in existed_strategies:
        new_numbers[strat] += adjustment_per_strategy
    
    # Convert the numbers back to frequencies
    for strat in new_numbers:
        current_frequencies[strat] = new_numbers[strat] / population_size
    
    adjusted_frequencies = adjust_remaining_frequencies(current_frequencies, strategy_to_adjust, new_frequency)
    
    return adjusted_frequencies

def update_population_with_changes(population, change, winners, strategy_frequencies):
    # If change is positive, add players
    if change > 0:
        # Add new players randomly based on the change
        new_players = [random.choice(list(strategy_frequencies.keys())) for _ in range(change)]
        print ("New Players:", new_players)
        population += new_players
        print(f"Added {change} new players to the population.")
    # If change is negative, remove players
    elif change < 0:
        # Remove players from the winners list
        random.shuffle(winners)
        removed_players = winners[:abs(change)]
        print ("Removed Players:",removed_players)
        for player in removed_players:
            population.remove(player)
        print(f"Removed {abs(change)} players from the winners list.")

    # Calculate and print the updated strategy frequencies after the changes
    updated_frequencies = calculate_new_frequencies(population)
    print(f"Updated strategy frequencies: {updated_frequencies}")
    
    return population

# Core Simulation Unit

def simulate_Metasimul8(strategy_frequencies, population_size, time_steps, matrix, 
                     disruption_time_payoff_matrix_change_list=None, 
                     Pay_off_matrix_change_list=None, 
                     new_strategy_name_list=None, 
                     disruption_matrices_new_strategies=None, 
                     disruption_time_new_strategy_list=None,
                     disrup_pop_time_list=None, disrup_pop_value=None, 
                     Strategy_Names_for_pop=None, New_strategy_freq_pop=None, 
                     distruption_time_strat_pop=None):
    

    # Initial Population Setup : Current Strategy Frequency
    population = []
    total_population_size = population_size
    for strategy, freq in strategy_frequencies.items():
        count = int(freq * population_size)
        population.extend([strategy] * count)

    results = []
    matches = []

    global_match_record = {}
    all_strategy_counts = {strategy: 0 for strategy in strategy_frequencies.keys()}
    

    # Check For Disruptions based on the Time Step (Pay Off Matric Change, New Strategy Disruption and Player Population)
    
    for step in range(time_steps):
        initial_strategy_counts = {strategy: population.count(strategy) for strategy in strategy_frequencies.keys()}
        disruption_type = None
        print(f"\nTime Step: {step}")
        for strategy, payoffs in matrix.items():
            print(f"{strategy}: {payoffs}")

        # Update the matrix if the current step is in disruption_times
        prev_matrix = matrix.copy()
        matrix = update_payoff_matrix(step, matrix, disruption_time_payoff_matrix_change_list, Pay_off_matrix_change_list)
        if matrix != prev_matrix:
            disruption_type = "Pay-off Matrix Change disruption"
        
        if step in disruption_time_new_strategy_list:
            disruption_type = "New Strategy Disruption"
            index = disruption_time_new_strategy_list.index(step)
            new_strategy_name = new_strategy_name_list[index]
            new_strategy_values = disruption_matrices_new_strategies[index]  # Get the matrix values
            strategy_frequencies, matrix = add_new_strategy(strategy_frequencies, matrix, new_strategy_name, new_strategy_values)
            print(f"New strategy '{new_strategy_name}' introduced at time step {step}.")
            print(f"Current Payoff Matrix: {matrix}")
        
        a = strategy_frequencies
      
        change = 0 
        if step in disrup_pop_time_list:
            disruption_type = "Player Popularity Disruption"
            change = disrup_pop_value[disrup_pop_time_list.index(step)]
            # Passing strategy_frequencies as an additional argument
            population = update_population_with_changes(population, change, winners, strategy_frequencies)
            
        # Adjust the loop to use the actual population size at each step
        current_population_size = len(population)

        # Shuffle the population for random pairing
        random.shuffle(population)

        # Pair Wise Interactions
        winners = []
        losers = []
        opponents = []

        # Iterate over pairs in the population
        for i in range(0, current_population_size - (current_population_size % 2), 2):
            player1, player2 = population[i], population[i + 1]
            
            # Determine the outcomes based on the payoff matrix
            player1_outcome = matrix[player1][player2]
            player2_outcome = matrix[player2][player1]
            
            # Determine winners and losers
            if player1_outcome > player2_outcome:
                winners.append(player1)
                losers.append(player2)
                opponents.append(player1)
                matches.append((player1, player2))
                global_match_record[(player1, player2)] = global_match_record.get((player1, player2), 0) + 1
                
            elif player1_outcome < player2_outcome:
                winners.append(player2)
                losers.append(player1)
                opponents.append(player2)
                matches.append((player2, player1))
                global_match_record[(player2, player1)] = global_match_record.get((player2, player1), 0) + 1
               
            else:  # It's a tie
                winners.extend([player1, player2])

        print(f"\nTime Step: {step}")
        print(f"Winners: {len(winners)}")
        print(f"Losers: {len(losers)}")
        print(f"Winner: {winners}")
        print(f"Losers before strategy selection: {losers}")


        expected_payoffs_at_step = calculate_expected_payoff(strategy_frequencies, matrix)
        
        
        win_percentages = calculate_win_percentage(matches)
        for match, result in win_percentages.items():
            print(f"At time step {step}, for {match}: {result}")

        win_percentages = calculate_overall_win_percentages(global_match_record)
        for match, result in win_percentages.items():
            print(match, ":", result)

        strategy_counts_winner = {strategy: sum(1 for a in winners if a == strategy) for strategy in set(winners)}
        winners_before = [f"{strategy}: {count}" for strategy, count in strategy_counts_winner.items()]
        print (winners_before)

        win_1 = {strategy: winners.count(strategy) for strategy in strategy_frequencies}

        strategy_counts_loser = {strategy: sum(1 for s in losers if s == strategy) for strategy in set(losers)}
        losers_before = [f"{strategy}: {count}" for strategy, count in strategy_counts_loser.items()]
        print (losers_before)

        lose_1= {strategy: losers.count(strategy) for strategy in strategy_frequencies}


        # Strategy selection for the next round
        
        # 60% Replicator
        replicator_count = int(len(losers) * 0.60)
        print(f"Replicators out of losers: {replicator_count}")
        
        selected_indices = random.sample(range(len(losers)), replicator_count)
        
        replicators_before = [losers[i] for i in selected_indices]
        print(f"Replicators before change: {replicators_before}")

        # Update the strategies of the selected replicators
        for index in selected_indices:
            losers[index] = opponents[index]

        replicators_after = [losers[i] for i in selected_indices]
        print(f"Replicators after change: {replicators_after}")
        print(f"No. of Replicators after change: {len(replicators_after)}")


        # 25% Best Response
        remaining_losers = list(set(range(len(losers))) - set(selected_indices))
        print("remaining losers:",len(remaining_losers))
        best_response_count = int(len(losers) * 0.25)
        print("best response count:",best_response_count)
        best_response_indices = random.sample(remaining_losers, best_response_count)
        print("best response indices:",len(best_response_indices))

        best_response_before = []
        best_response_after = []
        for index in best_response_indices:
            loser_strategy = losers[index]
            best_response_before.append(loser_strategy)
            
            # Get the last opponent's strategy for this loser
            opponent_strategy = opponents[index]
            # Print the opponent's strategy for clarity
            print(f"For loser {loser_strategy}, the last opponent's strategy was: {opponent_strategy}")
            
            # Find the best response against this opponent strategy
            response_strategy = best_response_against(opponent_strategy, matrix)
            losers[index] = response_strategy
            best_response_after.append(response_strategy)

        print(f"Best Response before change: {best_response_before}")
        print(f"Best Response after change: {best_response_after}")

       # BNN strategy selection: 10% losers adopt the strategy with the highest expected payoff
        best_strategy = max(expected_payoffs_at_step, key=expected_payoffs_at_step.get)

        remaining_losers_after_br_and_replicator = list(set(remaining_losers) - set(best_response_indices))
        print("remaining losers after br and rep:", len(remaining_losers_after_br_and_replicator))
        bnn_count = int(len(losers) * 0.10)
        print("BNN count", bnn_count)
        bnn_indices = random.sample(remaining_losers_after_br_and_replicator, bnn_count)
        print("BNN indices", len(bnn_indices))
        
        bnn_before = [losers[i] for i in bnn_indices]
        for index in bnn_indices:
            losers[index] = best_strategy
        bnn_after = [losers[i] for i in bnn_indices]

        print(f"BNN choices before change: {bnn_before}")
        print(f"BNN choices after change: {bnn_after}")

        # 5% Random strategy selection
        remaining_losers_after_bnn = list(set(remaining_losers_after_br_and_replicator) - set(bnn_indices))
        print("remaining losers after br, rep and bnn:", len(remaining_losers_after_bnn))
        random_before = []
        random_after = []
        strategies = list(strategy_frequencies.keys())
        for index in remaining_losers_after_bnn:
            random_before.append(losers[index])
            losers[index] = random.choice(strategies)
            random_after.append(losers[index])

        print(f"Random choices before change: {random_before}")
        print(f"Random choices after change: {random_after}")

        # Update the population for the next round
        population = winners + losers

        # Quantification Unit - Update Strategy Frequency

        # Update strategy frequencies based on the outcomes
        all_strategies = winners + losers
        strategy_frequencies = calculate_new_frequencies(all_strategies)
        print(f"\nStrategy Frequencies for time step {step}: {strategy_frequencies}")

        # Strategy Popularity Disruption Check
        print(f"Invoking Strategy_Popularity for timestep: {step+1}")
        prev_strategy_frequencies = strategy_frequencies.copy()
        strategy_frequencies = Strategy_Popularity(step, Strategy_Names_for_pop, New_strategy_freq_pop, distruption_time_strat_pop, strategy_frequencies, population_size)
        print(f"\nNew Strategy Frequencies for time step due to strategy popularity {step+1}: {strategy_frequencies}")
        # Check if there was any change in the strategy frequencies due to Strategy_Popularity function
        if prev_strategy_frequencies != strategy_frequencies and disruption_type is None:
            disruption_type = "Strategy Frequency Disruption"
            
        # If no disruption detected by the end of the loop iteration, set it to None
        if disruption_type is None:
            disruption_type = "None"
        
        # Player Population Re-Check

        if step not in disrup_pop_time_list:
            population = []  
            for strategy, freq in strategy_frequencies.items():
                count = int(freq * current_population_size)
                population.extend([strategy] * count)
                print(f"Population size at time step {step}: {len(population)}")

                # Check if the regenerated population size matches the current population size
            difference = current_population_size - len(population)
            if difference > 0:
                most_common_strategy = max(strategy_frequencies, key=strategy_frequencies.get)
                population.extend([most_common_strategy] * difference)
            elif difference < 0:
                for _ in range(abs(difference)):
                    population.remove(random.choice(population))

       
        # Count the number of each strategy in winners and losers
        winner_counts = {strategy: winners.count(strategy) for strategy in strategy_frequencies}
        loser_counts = {strategy: losers.count(strategy) for strategy in strategy_frequencies}

        matches.clear()

        
        results.append({
            "time_step": step,
            "strategy_counts": initial_strategy_counts,
            "Expected Payoff": expected_payoffs_at_step,
            "Strategy Frequency":a,
            "winners":win_1,
            "losers": lose_1,
            "winner": winners_before,
            "loser": losers_before,
            "New Strategy Frequency": strategy_frequencies,
            "wins": winner_counts,
            "loses":loser_counts,
            "disruption": disruption_type
            })

    return results, global_match_record

import tkinter as tk
import random

def display_graphical_representation_dynamic_v2(results, window_to_draw_on):
    # Extract all unique strategies from the results
    all_strategies = set()
    for timestep_data in results:
        current_payoff_matrix = timestep_data.get("Expected Payoff", {})
        all_strategies.update(current_payoff_matrix.keys())
        
    all_strategies = list(all_strategies)

    WIDTH, HEIGHT = 1200, 500
    SQUARE_SIZE = 200
    RADIUS = 5  # Size of circle representing an agent

    # Determine time per step for animation
    total_time_steps = len(results)
    time_per_step = int(10000 / total_time_steps)  # 10,000 ms (10 seconds) divided by the total time steps

    # Assign a unique color and position for each strategy
    strategy_colors = {strategy: "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for strategy in all_strategies}
    strategy_positions = {strategy: (WIDTH / (len(all_strategies) + 1) * (i+1), HEIGHT / 2) for i, strategy in enumerate(all_strategies)}

    current_time_step_idx = 0

    canvas = tk.Canvas(window_to_draw_on, width=WIDTH, height=HEIGHT)
    canvas.pack()

    # Create squares for strategies and add the strategy names below the squares
    for strategy, (x, y) in strategy_positions.items():
        canvas.create_rectangle(x - SQUARE_SIZE/2, y - SQUARE_SIZE/2, x + SQUARE_SIZE/2, y + SQUARE_SIZE/2, fill=strategy_colors[strategy], outline="black")
        # Add strategy name below the square
        canvas.create_text(x, y + SQUARE_SIZE/2 + 15, anchor='n', text=strategy, font=("Arial", 12, "bold"))

    # Placeholder for agent circles. Will be created dynamically in animate function.
    agents = []

    # Creating a text field to display disruption information
    disruption_text = canvas.create_text(WIDTH / 2, 20, text="", font=("Arial", 14, "bold"))

    def animate():
        nonlocal current_time_step_idx
        if current_time_step_idx < len(results):
            timestep_data = results[current_time_step_idx]

            # Display disruption information if there's any
            disruption_info = timestep_data.get("disruption", None)
            if disruption_info:
                canvas.itemconfig(disruption_text, text=f"Disruption: {disruption_info}")
            else:
                canvas.itemconfig(disruption_text, text="")
            
            # Calculate the current population size
            current_population_size = sum(timestep_data["wins"].values()) + sum(timestep_data["loses"].values())

            # Adjust the agent circles based on the current population size
            for agent in agents:
                canvas.delete(agent)
            agents.clear()
            agents.extend([canvas.create_oval(0, 0, RADIUS*2, RADIUS*2, fill="white") for _ in range(current_population_size)])

            # Update agents' positions and colors
            agent_idx = 0
            for strategy, count in timestep_data["wins"].items():
                for _ in range(count):
                    x, y = strategy_positions[strategy]
                    x += random.randint(-SQUARE_SIZE // 4, SQUARE_SIZE // 4)
                    y += random.randint(-SQUARE_SIZE // 4, SQUARE_SIZE // 4)
                    
                    canvas.coords(agents[agent_idx], x, y, x + RADIUS*2, y + RADIUS*2)
                    canvas.itemconfig(agents[agent_idx], fill=strategy_colors[strategy])
                    agent_idx += 1

            for strategy, count in timestep_data["loses"].items():
                for _ in range(count):
                    x, y = strategy_positions[strategy]
                    x += random.randint(-SQUARE_SIZE // 4, SQUARE_SIZE // 4)
                    y += random.randint(-SQUARE_SIZE // 4, SQUARE_SIZE // 4)
                    
                    canvas.coords(agents[agent_idx], x, y, x + RADIUS*2, y + RADIUS*2)
                    canvas.itemconfig(agents[agent_idx], fill=strategy_colors[strategy])
                    agent_idx += 1

            # Display strategy frequencies below the strategy name
            for strategy, freq in timestep_data["New Strategy Frequency"].items():
                x, y = strategy_positions[strategy]
                freq_text = f"Frequency: {freq:.2f}"
                canvas.delete(f"freq_{strategy}")
                canvas.create_text(x, y + SQUARE_SIZE/2 + 30, anchor='n', text=freq_text, font=("Arial", 10), tags=f"freq_{strategy}")

            # Display the current time step
            canvas.delete("current_time_step")
            canvas.create_text(WIDTH / 2, HEIGHT / 4, text=f"Time Step: {current_time_step_idx + 1}", font=("Arial", 14, "bold"), tags="current_time_step")

            current_time_step_idx += 1
            window_to_draw_on.after(time_per_step, animate)

    animate()
    

import tkinter as tk
import random


def pacman_dominance(results, canvas, parent):
    

    WIDTH, HEIGHT = 600, 400
    PACMAN_SIZE = 50
    CIRCLE_SIZE = 50
    EYE_SIZE = 5  # Size of the ghost eyes
    
    strategy_colors = {}
    strategy_circles = {}
    strategy_eyes = {}

    def reset_canvas(sorted_strategies, strategy_positions):
        canvas.delete("all")
        for strategy, (x, y) in strategy_positions.items():
            strategy_text = canvas.create_text(x, y + PACMAN_SIZE + 15, anchor='n', text=strategy, font=("Arial", 12, "bold"))
            circle = canvas.create_oval(x - CIRCLE_SIZE, y - CIRCLE_SIZE, x + CIRCLE_SIZE, y + CIRCLE_SIZE, fill=strategy_colors[strategy], outline=strategy_colors[strategy])
            
            # Adding eyes for the ghost look
            eye_y = y - EYE_SIZE
            left_eye_x = x - (CIRCLE_SIZE / 2)
            right_eye_x = x + (CIRCLE_SIZE / 2) - (2 * EYE_SIZE)
            left_eye = canvas.create_oval(left_eye_x, eye_y, left_eye_x + EYE_SIZE, eye_y + EYE_SIZE, fill="white", outline="white")
            right_eye = canvas.create_oval(right_eye_x, eye_y, right_eye_x + EYE_SIZE, eye_y + EYE_SIZE, fill="white", outline="white")
            
            strategy_circles[strategy] = circle
            strategy_eyes[strategy] = (left_eye, right_eye)


    def animate_pacman(timestep):
        timestep = timestep + 1 
        print(f"Animating for user-perceived time step: {timestep}")  # Adjusted print message
    
        if timestep < len(results):

            
            winners_at_timestep = results[timestep]['winners']
            losers_at_timestep = results[timestep]['losers']

            # Calculate net sum for each strategy
            net_sum = {strategy: winners_at_timestep.get(strategy, 0) - losers_at_timestep.get(strategy, 0) for strategy in set(winners_at_timestep) | set(losers_at_timestep)}

            # Sort the strategies based on their net sum in descending order
            sorted_strategies = sorted(net_sum, key=net_sum.get, reverse=True)
            
            # Update strategy colors
            for strategy in sorted_strategies:
                if strategy not in strategy_colors:
                    strategy_colors[strategy] = '#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            
            # Calculate strategy positions based on the sorted order
            strategy_positions = {strategy: (WIDTH / (len(sorted_strategies) + 1) * (i+1), HEIGHT / 2) 
                                  for i, strategy in enumerate(sorted_strategies)}

            reset_canvas(sorted_strategies, strategy_positions)
            
            for i in range(len(sorted_strategies) - 1, 0, -1):  # Start from the second last strategy
                dominant_strategy = sorted_strategies[i-1]
                weaker_strategy = sorted_strategies[i]
                
                dominant_x, dominant_y = strategy_positions[dominant_strategy]
                weaker_x, weaker_y = strategy_positions[weaker_strategy]
                
                dx = (weaker_x - dominant_x) / 10
                dy = (weaker_y - dominant_y) / 10
                
                start_angle, extent_angle = 30, 300
                pacman_color = strategy_colors[dominant_strategy]
                
                pacman = canvas.create_arc(dominant_x - PACMAN_SIZE, dominant_y - PACMAN_SIZE, 
                                           dominant_x + PACMAN_SIZE, dominant_y + PACMAN_SIZE, 
                                           start=start_angle, extent=extent_angle, 
                                           fill=pacman_color, outline=pacman_color)

                for j in range(10):
                    canvas.move(pacman, dx, dy)
                    parent.after(50)  # Added delay for animation
                    parent.update()  # Use 'parent' instead of 'root'
                    
                # "Engulf" the weaker strategy by removing its circle, eyes, and label
                canvas.delete(strategy_circles[weaker_strategy])
                canvas.delete(strategy_eyes[weaker_strategy][0])
                canvas.delete(strategy_eyes[weaker_strategy][1])
                canvas.delete(weaker_strategy)

                # Remove the old pacman
                canvas.delete(pacman)

    def get_timestep_and_animate():
        timestep = int(entry.get())
        animate_pacman(timestep - 1)  # Subtracting 1 to make it 0-indexed

    label = tk.Label(parent, text="Enter Time Step:")  # Use 'parent' instead of 'root'
    label.pack(pady=5)

    entry = tk.Entry(parent)  # Use 'parent' instead of 'root'
    entry.pack(pady=5)

    button = tk.Button(parent, text="Show Dominance", command=get_timestep_and_animate)  # Use 'parent' instead of 'root'
    button.pack(pady=5)

    reset_canvas({}, {})  # Empty initial setup


import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import inset_locator
from mpl_toolkits.axes_grid1 import make_axes_locatable

def show_strategy_plotter(parent, results, global_match_record):
    
    parent.title("Strategy Plotter")
    control_frame = ttk.Frame(parent)
    control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ns")

    plot_frame = ttk.Frame(parent)
    plot_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
    fig, ax = plt.subplots(figsize=(6, 4))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_expected_payoffs_heatmap(ax, results, time_steps):
        ax.clear()
        ax.set_position([0.1, 0.1, 0.85, 0.8])

        all_strategies = set()
        for r in results:
            all_strategies.update(r["Expected Payoff"].keys())
            
        expected_payoffs_data = {strategy: [] for strategy in all_strategies}


        for cbar in [c for c in ax.figure.get_axes() if isinstance(c, plt.Axes) and not c is ax]:
            cbar.remove()
        

        for r in results:
            for strategy in all_strategies:
                payoff = r["Expected Payoff"].get(strategy, np.nan)
                expected_payoffs_data[strategy].append(payoff)
                
            
        strategies = list(expected_payoffs_data.keys())
        data_for_heatmap = [expected_payoffs_data[strategy] for strategy in strategies]
    
        # Using imshow to plot the heatmap
        im = ax.imshow(data_for_heatmap, cmap="YlGnBu", aspect='auto')

        divider = make_axes_locatable(ax)
        axins = divider.append_axes("right", size="2%", pad=0.0)


        ax.set_yticks(np.arange(len(strategies)))
        ax.set_yticklabels(strategies)

        # Setting the x-ticks to be the time steps (you can customize this for clarity)
        ax.set_xticks(np.arange(0, len(time_steps), len(time_steps)//5))
        ax.set_xticklabels(time_steps[::len(time_steps)//5])

        # Displaying a colorbar
        cbar = ax.figure.colorbar(im, cax=axins, label='Expected Payoff', orientation='vertical')
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Strategies")
        ax.set_title("Expected Payoffs Over Time")
        plt.tight_layout()

    def moving_average(data, window_size):
        """Compute moving average."""
        cumsum = [0]
        moving_aves = []
        for i, x in enumerate(data, 1):
            cumsum.append(cumsum[i-1] + x)
            if i >= window_size:
                moving_ave = (cumsum[i] - cumsum[i-window_size]) / window_size
                moving_aves.append(moving_ave)
        return moving_aves

    def plot_results(results, selected_strategies, option, smooth=False):
        time_steps = [r['time_step'] for r in results]
        ax.clear()
        ax.axis('on')

        for cbar in [c for c in ax.figure.get_axes() if isinstance(c, plt.Axes) and not c is ax]:
            cbar.remove()
    
        if option == "Strategy Frequency vs time":
            for strategy in selected_strategies:
                freqs = [r['Strategy Frequency'].get(strategy, 0) for r in results]
                if smooth:
                    smoothed_freqs = moving_average(freqs, window_size=5)
                    ax.plot(time_steps[len(time_steps)-len(smoothed_freqs):], smoothed_freqs, label=strategy)
                else:
                    ax.plot(time_steps, freqs, label=strategy)
            ax.set_ylabel('Strategy Frequency')
            ax.set_xlabel('Time Step')
            ax.set_title('Selected Strategy Frequency Over Time' + (' (Smoothed)' if smooth else ''))
            ax.legend()

        elif option == "Population vs time":
            for strategy in selected_strategies:
                strategy_counts = [r['strategy_counts'].get(strategy, 0) for r in results]
                if smooth:
                    smoothed_counts = moving_average(strategy_counts, window_size=5)
                    ax.plot(time_steps[len(time_steps)-len(smoothed_counts):], smoothed_counts, label=strategy)
                else:
                    ax.plot(time_steps, strategy_counts, label=strategy)
            ax.set_ylabel('Strategy Count')
            ax.set_xlabel('Time Step')
            ax.set_title('Population of Each Strategy Over Time Steps' + (' (Smoothed)' if smooth else ''))
            ax.legend()
            
        elif option == "Pair-Wise Interaction vs Time":
            for strategy in selected_strategies:
                winner_counts = []
                for r in results:
                    win_dict = {x.split(": ")[0]: int(x.split(": ")[1]) for x in r['winner']}
                    winner_counts.append(win_dict.get(strategy, 0))
                ax.plot(time_steps, winner_counts, label=strategy + " - Winners")
            

                loser_counts = []
                
                for r in results:
                    lose_dict = {x.split(": ")[0]: int(x.split(": ")[1]) for x in r['loser']}
                    loser_counts.append(lose_dict.get(strategy, 0))
                ax.plot(time_steps, loser_counts, label=strategy + " - Losers")
                
                
            ax.set_ylabel('Pair-Wise Interaction (Before Selection)')
            ax.set_xlabel('Time Step')
            ax.set_title('Strategy Counts Before Selection (Separate Winners and Losers)')
            ax.legend()

        elif option == "After Strategy Selection vs Time":
            for strategy in selected_strategies:
                after_counts = []
                for r in results:
                    wins_dict = r['wins']
                    loses_dict = r['loses']
                    after_counts.append(wins_dict.get(strategy, 0) + loses_dict.get(strategy, 0))
                ax.plot(time_steps, after_counts, label=strategy)
            ax.set_ylabel('Strategy Count After Selection')
            ax.set_xlabel('Time Step')
            ax.set_title('Strategy Counts After Selection')
            ax.legend()

        elif option == "Expected pay-offs vs time":
            plot_expected_payoffs_heatmap(ax, results, time_steps)
            

        elif option == "Strategy Frequency vs Expected Pay-off":
            for strategy in selected_strategies:
                freqs = [r['Strategy Frequency'].get(strategy, 0) for r in results]
                expected_payoffs = [r['Expected Payoff'].get(strategy, 0) for r in results]
                ax.scatter(freqs, expected_payoffs, label=strategy)
            ax.set_xlabel('Strategy Frequency')
            ax.set_ylabel('Expected Payoff')
            ax.set_title('Strategy Frequency vs Expected Pay-off')
            ax.legend()

    
        else:
            ax.set_xlabel('Time Step')
            ax.legend()
        canvas.draw()

    def update_plot_based_on_dropdown():
        selected_strategies = [strat.get() for strat in strategy_vars if strat.get()]
        selected_option = dropdown_var.get()
        plot_results(results, selected_strategies, selected_option)

    def smooth_plot():
        selected_strategies = [strat.get() for strat in strategy_vars if strat.get()]
        selected_option = dropdown_var.get()
        plot_results(results, selected_strategies, selected_option, smooth=True)


    def plot_disruptions_on_strategy_plot(results):
        abbreviations = {
            "New Strategy Disruption": "NSW",
            "Pay-off Matrix Change disruption": "POMC",
            "Player Popularity Disruption": "PP",
            "Strategy Frequency Disruption": "SF"
        }
        
        unique_disruptions = set(r['disruption'] for r in results if r['disruption'] != "None")
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_disruptions)))
        color_map = dict(zip(unique_disruptions, colors))
        for res in results:
            disruption = res['disruption']
            if disruption != "None":
                ax.plot(res['time_step'], 0, 'o', color=color_map[disruption])
                ax.text(res['time_step'], 0.08, abbreviations[disruption], fontsize=10, ha='center', va='bottom', color=color_map[disruption], rotation=90)

    def disruption_indicator():
        plot_disruptions_on_strategy_plot(results)
        canvas.draw() 

    def calculate_matchup_percentages1(global_match_record):
        strategies = list(set([matchup[0] for matchup in global_match_record.keys()] + [matchup[1] for matchup in global_match_record.keys()]))
        df = pd.DataFrame(0, index=strategies, columns=strategies, dtype=float)
        for matchup, count in global_match_record.items():
            strategy_a, strategy_b = matchup
            reverse_matchup_count = global_match_record.get((strategy_b, strategy_a), 0)
            if count + reverse_matchup_count > 0:
                win_percentage = (count / (count + reverse_matchup_count)) * 100
            else:
                win_percentage = 0
            df.at[strategy_a, strategy_b] = win_percentage
        for strategy in df.index:
            df.at[strategy, strategy] = '/'
        pd.options.display.float_format = '{:,.0f}'.format
        return df

    def display_matchup_table(control_frame):
        #global control_frame
        table_frame = ttk.LabelFrame(control_frame, text="Matchup Table", padding=(10, 5))
        table_frame.pack(pady=10, padx=10, fill="x")
        matchup_table = calculate_matchup_percentages1(global_match_record)
        matchup_table['Strategies'] = matchup_table.index
        ordered_columns = ['Strategies'] + [col for col in matchup_table if col != 'Strategies']
        matchup_table = matchup_table[ordered_columns]

        tree = ttk.Treeview(table_frame, columns=ordered_columns, show="headings")
        tree.pack(expand=True, fill="both", padx=5, pady=5)
        for col in ordered_columns:
            tree.heading(col, text=col)
            tree.column(col, width=60, anchor="center")
        for index, row in matchup_table.iterrows():
            tree.insert("", "end", values=tuple(row))
    
    
    plot_frame = ttk.Frame(parent)
    plot_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
    fig, ax = plt.subplots(figsize=(6, 4))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    plt.close(fig)
    control_frame = ttk.Frame(parent)
    control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ns")
    strategy_frame = ttk.LabelFrame(control_frame, text="Select Strategies", padding=(10, 5))
    all_strategies = set()
    for res in results:
        all_strategies.update(res['New Strategy Frequency'].keys())
        all_strategies.update(res['strategy_counts'].keys())
    strategy_vars = []
    for strategy in all_strategies:
        strategy_var = tk.StringVar(value=strategy)
        strategy_vars.append(strategy_var)
        ttk.Checkbutton(strategy_frame, text=strategy, variable=strategy_var, onvalue=strategy, offvalue="").pack(anchor='w', padx=10, pady=2, side=tk.LEFT)
    display_matchup_table(control_frame)
    strategy_frame.pack(pady=10, padx=10, fill="x")
    options = [
        "Strategy Frequency vs time",
        "Population vs time",
        "Pair-Wise Interaction vs Time",
        "After Strategy Selection vs Time",
        "Expected pay-offs vs time",
        "Strategy Frequency vs Expected Pay-off"]


    dropdown_var = tk.StringVar()
    dropdown = ttk.Combobox(control_frame, textvariable=dropdown_var, values=options, width=30)
    dropdown.pack(pady=10, padx=20, fill="x")
    dropdown_var.set(options[0])

    ttk.Button(control_frame, text="Update Plot", command=update_plot_based_on_dropdown).pack(pady=10)
    ttk.Button(control_frame, text="Smooth the Plot", command=smooth_plot).pack(pady=10)
    ttk.Button(control_frame, text="Disruption Indicator", command=disruption_indicator).pack(pady=10)

    parent.mainloop()


