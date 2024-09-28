from scipy.integrate import RK45
import matplotlib.pyplot as plt

def sis_model(t, y):
    # Parameters of the disease
    beta = 0.8  # Infection rate
    alpha = 0.1  # Recovery rate

    # y needs to be packaged in a list for the solver to understand it
    S, I = y

    dI_dt = beta * S * I / N - alpha * I
    dS_dt = -beta * S * I / N + alpha * I
    return dS_dt, dI_dt

def solve_model_RK45(model, N, I0, T):

    S0 = N - I0  # Number of original susceptible

    y0 = [S0, I0]

    # Initialising the lists that store the results
    times=[]
    S_vals=[]
    I_vals=[]

    solver = RK45(model, 0, y0=y0,t_bound= T) # initialising the Runge-Kutta solver

    while solver.status == 'running': # needed because apparently this solver doesn't run itself
        solver.step()

        # Store the results at each time step
        times.append(solver.t)
        S,I = solver.y
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
    plt.title('SIS Model Using RK45')
    plt.show()

if __name__ == '__main__':
    N = 10000  # Number of habitants (of only one city for now)
    I0 = 10  # Number of original infected
    T = 50 # Number of iterations

    sis_times, sis_S, sis_I = solve_model_RK45(sis_model, N, I0, T)

    plot_SIR(sis_times, sis_S, sis_I)