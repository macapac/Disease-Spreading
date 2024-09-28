from scipy.integrate import RK45
import matplotlib.pyplot as plt

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

def solve_model_RK45(model, N, I0, T, include_R=False):

    S0 = N - I0  # Number of original susceptible

    max_stepsize = 0.1

    # Initialising the lists that store the results
    times = []
    S_vals = []
    I_vals = []

    if include_R:
        y0 = [S0, I0, 0]
        R_vals = []

    else:
        y0 = [S0, I0]

    solver = RK45(model, 0, y0=y0, t_bound=T, max_step=max_stepsize)  # initialising the Runge-Kutta solver

    if include_R:
        while solver.status == 'running':  # needed because apparently this solver doesn't run itself
            solver.step()

            # Store the results at each time step
            times.append(solver.t)
            S, I, R = solver.y
            S_vals.append(S)
            I_vals.append(I)
            R_vals.append(R)

        return times, S_vals, I_vals, R_vals


    else:
        while solver.status == 'running':  # needed because apparently this solver doesn't run itself
            solver.step()

            # Store the results at each time step
            times.append(solver.t)
            S, I = solver.y
            S_vals.append(S)
            I_vals.append(I)

        return times, S_vals, I_vals

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

if __name__ == '__main__':
    N = 10000  # Number of inhabitants (of only one city for now)
    I0 = 10  # Number of original infected
    T = 50 # Number of iterations

    sis_times, sis_S, sis_I = solve_model_RK45(sis_model, N, I0, T)

    plot_SIR(sis_times, sis_S, sis_I)

    sir_times, sir_S, sir_I, sir_R = solve_model_RK45(sir_model, N, I0, T, include_R=True)

    plot_SIR(sir_times, sir_S, sir_I, sir_R)