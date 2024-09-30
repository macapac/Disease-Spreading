from scipy.integrate import RK45
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


if __name__ == '__main__':
    N = 10000  # Number of inhabitants (of only one city for now)
    I0 = 10  # Number of original infected
    T = 80 # Number of iterations

    sis_times, sis_S, sis_I = solve_model_RK45(sis_model, N, I0, T, include_R=False)

    plot_SIR(sis_times, sis_S, sis_I)

    sir_times, sir_S, sir_I, sir_R = solve_model_RK45(sir_model, N, I0, T, include_R=True)

    plot_SIR(sir_times, sir_S, sir_I, sir_R)

    # plot of cities
    times, city1_data, city2_data, city3_data = solve_model_RK45(sir_with_mobility, N, I0, T, include_R=True, cities=3)

    plot_SIR_multiple_cities(times, city1_data, city2_data, city3_data)

