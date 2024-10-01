from scipy.integrate import RK45
from scipy.integrate import RK23
from scipy.integrate import DOP853
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

def sis_model(t, y):
    # Parameters of the disease
    beta = 0.5  # Infection rate
    alpha = 0.2  # Recovery rate

    # y needs to be packaged in a list for the solver to understand it
    S, I = y

    dI_dt = beta * S * I / N - alpha * I
    dS_dt = -beta * S * I / N + alpha * I
    return dS_dt, dI_dt

def sir_model(t, y):
    # Parameters of the disease
    beta = 0.8  # Infection rate
    alpha = 0.1 # Recovery rate
    gamma = 0.005  # Vaccination rate

    # y contains the state variables: [S, I, R]
    S, I, R = y

    # Population size (assumed constant)
    N = S + I + R

    # Differential equations
    dS_dt = (-beta * S * I / N) - gamma * S  # Susceptible individuals becoming infected
    dI_dt = (beta * S * I / N) - alpha * I  # Infected individuals either recovering or infecting others
    dR_dt = gamma * S + alpha * I  # Recovered individuals

    return dS_dt, dI_dt, dR_dt

def sir_with_mobility(t, y):
    beta = 0.6  # Infection rate
    alpha = 0.05  # Recovery rate
    gamma = 0.005  # Vaccination rate
    LOCKDOWN_THRESHOLD = 1.10  # 10% infection threshold for lockdown

    # y = [S1, I1, R1, S2, I2, R2, S3, I3, R3]
    S1, I1, R1, S2, I2, R2, S3, I3, R3 = y

    N1 = S1 + I1 + R1
    N2 = S2 + I2 + R2
    N3 = S3 + I3 + R3

    # Infection rates and recovery for each city
    dS1_dt = (-beta * S1 * I1 / N1) - gamma * S1  # Susceptible in city 1
    dI1_dt = (beta * S1 * I1 / N1) - alpha * I1  # Infected in city 1
    dR1_dt = gamma * S1 + alpha * I1  # Recovered in city 1

    dS2_dt = (-beta * S2 * I2 / N2) - gamma * S2  # Susceptible in city 2
    dI2_dt = (beta * S2 * I2 / N2) - alpha * I2  # Infected in city 2
    dR2_dt = gamma * S2 + alpha * I2  # Recovered in city 2

    dS3_dt = (-beta * S3 * I3 / N3) - gamma * S3  # Susceptible in city 3
    dI3_dt = (beta * S3 * I3 / N3) - alpha * I3  # Infected in city 3
    dR3_dt = gamma * S3 + alpha * I3  # Recovered in city 3

    if t % 2 == 0:  # (morning, 0-12 hours)
        travel_rates = np.array([[0, 0.01, 0.05],  # to City 1
                                [0.005, 0, 0.05],  # to City 2
                                [0.005, 0.0, 0]])  # to City 3
    else:  # (evening, 12-24 hours)
        travel_rates = np.array([[0, 0.005, 0.005],  # to City 1
                                 [0.1, 0, 0.0],  # to City 2
                                 [0.05, 0.05, 0]])

    if I1 / N1 > LOCKDOWN_THRESHOLD:
        travel_rates[0, :] = 0  # City 1 lockdown: no one leaves or enters
        travel_rates[:, 0] = 0  # No one can enter city 1

    if I2 / N2 > LOCKDOWN_THRESHOLD:
        travel_rates[1, :] = 0  # City 2 lockdown: no one leaves or enters
        travel_rates[:, 1] = 0  # No one can enter city 2

    if I3 / N3 > LOCKDOWN_THRESHOLD:
        travel_rates[2, :] = 0  # City 3 lockdown: no one leaves or enters
        travel_rates[:, 2] = 0  # No one can enter city

    # Movement between cities (travel_rates define movement proportions between cities)
    # This models the exchange of populations between cities
    dS1_dt += travel_rates[1][0] * S2 - travel_rates[0][1] * S1  # Movement between city 1 and city 2
    dS1_dt += travel_rates[2][0] * S3 - travel_rates[0][2] * S1  # Movement between city 1 and city 3
    dS2_dt += travel_rates[0][1] * S1 - travel_rates[1][0] * S2  # Movement between city 2 and city 1
    dS2_dt += travel_rates[2][1] * S3 - travel_rates[1][2] * S2  # Movement between city 2 and city 3
    dS3_dt += travel_rates[0][2] * S1 - travel_rates[2][0] * S3  # Movement between city 3 and city 1
    dS3_dt += travel_rates[1][2] * S2 - travel_rates[2][1] * S3  # Movement between city 3 and city 2

    return [dS1_dt, dI1_dt, dR1_dt, dS2_dt, dI2_dt, dR2_dt, dS3_dt, dI3_dt, dR3_dt]

def solve_model_RK45(model, N, I0, T, include_R=False, cities=1):
    if cities == 1:
        S0 = N - I0  # Number of original susceptible
        if include_R:
            y0 = [S0, I0, 0]  # Initial state (S, I, R)
        else:
            y0 = [S0, I0]  # Initial state (S, I)
    else:
        # For the sir_with_mobility model, initial states for multiple cities
        S1, S2, S3 = (N - I0) // 3, (N - I0) // 3, (N - I0) // 3  # Dividing population across cities
        I1, I2, I3 = I0 // 3, I0 // 3, I0 // 3
        y0 = [S1, I1, 0, S2, I2, 0, S3, I3, 0]  # Initial state for multiple cities

    max_stepsize = 0.1

    # Initialize lists to store results
    times = []
    if cities == 1:
        S_vals = []
        I_vals = []
        R_vals = [] if include_R else None
    else:
        S1_vals, I1_vals, R1_vals = [], [], []
        S2_vals, I2_vals, R2_vals = [], [], []
        S3_vals, I3_vals, R3_vals = [], [], []

    # Initialize the solver
    solver = RK45(model, 0, y0=y0, t_bound=T, max_step=max_stepsize)

    while solver.status == 'running':
        solver.step()
        times.append(solver.t)
        if cities == 1:
            if include_R:
                S, I, R = solver.y
                S_vals.append(S)
                I_vals.append(I)
                R_vals.append(R)
            else:
                S, I = solver.y
                S_vals.append(S)
                I_vals.append(I)
        else:
            S1, I1, R1, S2, I2, R2, S3, I3, R3 = solver.y
            S1_vals.append(S1)
            I1_vals.append(I1)
            R1_vals.append(R1)
            S2_vals.append(S2)
            I2_vals.append(I2)
            R2_vals.append(R2)
            S3_vals.append(S3)
            I3_vals.append(I3)
            R3_vals.append(R3)

    if cities == 1:
        # return times, S_vals, I_vals, R_vals if include_R else None
        if include_R:
            return times, S_vals, I_vals, R_vals
        else:
            return times, S_vals, I_vals
    else:
        return times, (S1_vals, I1_vals, R1_vals), (S2_vals, I2_vals, R2_vals), (S3_vals, I3_vals, R3_vals)

def solve_model(model, N, I0, T, include_R=False, cities=1, solver=None):
    if cities == 1:
        S0 = N - I0  # Number of original susceptible
        if include_R:
            y0 = [S0, I0, 0]  # Initial state (S, I, R)
        else:
            y0 = [S0, I0]  # Initial state (S, I)
    else:
        # For the sir_with_mobility model, initial states for multiple cities
        S1, S2, S3 = (N - I0) // 3, (N - I0) // 3, (N - I0) // 3  # Dividing population across cities
        I1, I2, I3 = I0 // 3, I0 // 3, I0 // 3
        y0 = [S1, I1, 0, S2, I2, 0, S3, I3, 0]  # Initial state for multiple cities

    max_stepsize = 0.01

    # Initialize lists to store results
    times = []
    if cities == 1:
        S_vals = []
        I_vals = []
        R_vals = [] if include_R else None
    else:
        S1_vals, I1_vals, R1_vals = [], [], []
        S2_vals, I2_vals, R2_vals = [], [], []
        S3_vals, I3_vals, R3_vals = [], [], []

    # Initialize the solver
    if solver == 'RK45':
        solver = RK45(model, 0, y0=y0, t_bound=T, max_step=max_stepsize)
    elif solver == 'DOP853':
        solver = DOP853(model, 0, y0=y0, t_bound=T, max_step=max_stepsize)
    else:
        solver = RK23(model, 0, y0=y0, t_bound=T, max_step=max_stepsize)


    while solver.status == 'running':
        solver.step()
        times.append(solver.t)
        if cities == 1:
            if include_R:
                S, I, R = solver.y
                S_vals.append(S)
                I_vals.append(I)
                R_vals.append(R)
            else:
                S, I = solver.y
                S_vals.append(S)
                I_vals.append(I)
        else:
            S1, I1, R1, S2, I2, R2, S3, I3, R3 = solver.y
            S1_vals.append(S1)
            I1_vals.append(I1)
            R1_vals.append(R1)
            S2_vals.append(S2)
            I2_vals.append(I2)
            R2_vals.append(R2)
            S3_vals.append(S3)
            I3_vals.append(I3)
            R3_vals.append(R3)

    if cities == 1:
        # return times, S_vals, I_vals, R_vals if include_R else None
        if include_R:
            return times, S_vals, I_vals, R_vals
        else:
            return times, S_vals, I_vals
    else:
        return times, (S1_vals, I1_vals, R1_vals), (S2_vals, I2_vals, R2_vals), (S3_vals, I3_vals, R3_vals)

def plot_SIR(times, S_vals, I_vals, R_vals = None):

    plt.plot(times, S_vals, label='Susceptible')
    plt.plot(times, I_vals, label='Infected', color='red')
    if R_vals is not None:
        plt.plot(times, R_vals, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    if R_vals:
        plt.title('SIR Model Using RK45')
    else:
        plt.title('SIS Model Using RK45')
    plt.show()

def plot_SIR_multiple_cities(times, city1_data, city2_data, city3_data):
    S1_vals, I1_vals, R1_vals = city1_data
    S2_vals, I2_vals, R2_vals = city2_data
    S3_vals, I3_vals, R3_vals = city3_data

    plt.figure(figsize=(10, 6))

    # Plot for City 1
    plt.plot(times, S1_vals, label='City 1 Susceptible', color='red')
    plt.plot(times, I1_vals, '--', label='City 1 Infected', color='red')
    plt.plot(times, R1_vals, ':', label='City 1 Recovered', color='red')

    # Plot for City 2
    plt.plot(times, S2_vals, label='City 2 Susceptible', color='green')
    plt.plot(times, I2_vals, '--', label='City 2 Infected', color='green')
    plt.plot(times, R2_vals, ':', label='City 2 Recovered', color='green')

    # Plot for City 3
    plt.plot(times, S3_vals, label='City 3 Susceptible', color='blue')
    plt.plot(times, I3_vals, '--', label='City 3 Infected', color='blue')
    plt.plot(times, R3_vals, ':', label='City 3 Recovered', color='blue')

    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('SIR Model with Mobility Across Three Cities')
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.show()

def animate_SIR_multiple_cities(times, city1_data, city2_data, city3_data):
    def round_nested_tuple(tuple_of_lists): # Quick helper function to round to intigers
        return tuple([
            [round(num) for num in inner_list]
            for inner_list in tuple_of_lists
        ])

    S1_vals, I1_vals, R1_vals = round_nested_tuple(city1_data)
    S2_vals, I2_vals, R2_vals = round_nested_tuple(city2_data)
    S3_vals, I3_vals, R3_vals = round_nested_tuple(city3_data)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set limits for city boundaries
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    ax.set_axis_off()
    # Define colors for S, I, and R
    S_color = 'blue'
    I_color = 'red'
    R_color = 'green'

    # Function to precompute the grid positions
    def place_in_grid(max_people, x_start, x_end, y_start, y_end):
        """Place dots in a grid layout within the given city bounds."""
        rows = int(np.sqrt(max_people))  # Create a square-like grid
        cols = (max_people // rows) + 1
        x_positions = np.linspace(x_start + 0.1, x_end - 0.1, cols)
        y_positions = np.linspace(y_start + 0.1, y_end - 0.1, rows)
        grid_x, grid_y = np.meshgrid(x_positions, y_positions)
        return grid_x.flatten(), grid_y.flatten()

    # Precompute maximum population
    max_num = max(max(max(S1_vals),max(S2_vals),max(S3_vals)), max(max(I1_vals),max(I2_vals),max(I3_vals)), max(max(R1_vals),max(R2_vals),max(R3_vals)))

    x1_S, y1_S = place_in_grid(max_num, 0, 1, 2, 3)
    x1_I, y1_I = place_in_grid(max_num, 0, 1, 1, 2)
    x1_R, y1_R = place_in_grid(max_num, 0, 1, 0, 1)

    x2_S, y2_S = place_in_grid(max_num, 1, 2, 2, 3)
    x2_I, y2_I = place_in_grid(max_num, 1, 2, 1, 2)
    x2_R, y2_R = place_in_grid(max_num, 1, 2, 0, 1)

    x3_S, y3_S = place_in_grid(max_num, 2, 3, 2, 3)
    x3_I, y3_I = place_in_grid(max_num, 2, 3, 1, 2)
    x3_R, y3_R = place_in_grid(max_num, 2, 3, 0, 1)

    # Update function for animation
    def update(frame):
        ax.clear()

        # Set limits for city boundaries
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_axis_off()

        # Draw city boundaries
        ax.plot([1, 1], [0, 3], color='black', linewidth=1.5)  # Boundary between city 1 and city 2
        ax.plot([2, 2], [0, 3], color='black', linewidth=1.5)  # Boundary between city 2 and city 3

        # Get current population values
        S1, I1, R1 = S1_vals[frame], I1_vals[frame], R1_vals[frame]
        S2, I2, R2 = S2_vals[frame], I2_vals[frame], R2_vals[frame]
        S3, I3, R3 = S3_vals[frame], I3_vals[frame], R3_vals[frame]

        # Plot City 1
        ax.scatter(x1_S[:S1], y1_S[:S1], color=S_color, s=10)
        ax.scatter(x1_I[:I1], y1_I[:I1], color=I_color, s=10)
        ax.scatter(x1_R[:R1], y1_R[:R1], color=R_color, s=10)

        # Plot City 2
        ax.scatter(x2_S[:S2], y2_S[:S2], color=S_color, s=10)
        ax.scatter(x2_I[:I2], y2_I[:I2], color=I_color, s=10)
        ax.scatter(x2_R[:R2], y2_R[:R2], color=R_color, s=10)

        # Plot City 3
        ax.scatter(x3_S[:S3], y3_S[:S3], color=S_color, s=10)
        ax.scatter(x3_I[:I3], y3_I[:I3], color=I_color, s=10)
        ax.scatter(x3_R[:R3], y3_R[:R3], color=R_color, s=10)

        # Add titles and labels
        ax.set_title(f'Time: {times[frame]:.2f}')

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(times), interval=100, repeat=False)

    # Show the plot
    plt.show()

if __name__ == '__main__':
    N = 1000  # Number of inhabitants
    I0 = 10  # Number of original infected
    T = 80 # Number of iterations

    '''sis_times, sis_S, sis_I = solve_model_RK45(sis_model, N, I0, T, include_R=False)

    plot_SIR(sis_times, sis_S, sis_I)

    sir_times, sir_S, sir_I, sir_R = solve_model_RK45(sir_model, N, I0, T, include_R=True)

    plot_SIR(sir_times, sir_S, sir_I, sir_R)'''

    # plot of cities
    # use solver ='DOP853' or ='RK45' or ='RK23'
    times, city1_data, city2_data, city3_data = solve_model(sir_with_mobility, N, I0, T, include_R=True, cities=3, solver='DOP853')

    plot_SIR_multiple_cities(times, city1_data, city2_data, city3_data)

    animate_SIR_multiple_cities(times, city1_data, city2_data, city3_data)