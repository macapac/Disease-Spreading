from scipy.integrate import RK45
import matplotlib.pyplot as plt

N = 10000  # Number of habitants (of only one city for now)
I0 = 10  # Number of original infected
S0 = N - I0  # Number of original susceptible

y0 = [S0, I0]

# Parameters of the disease
beta = 0.8  # Infection rate
alpha = 0.1  # Recovery rate

T = 50 # Max time
time = range(0,T)

def sis_model(t, y):
    # y needs to be packaged in a list for the solver to understand it
    S, I = y

    dI_dt = beta * S * I / N - alpha * I
    dS_dt = -beta * S * I / N + alpha * I
    return dS_dt, dI_dt

# Initialising the lists that store the results
times=[]
S_vals=[]
I_vals=[]

solver = RK45(sis_model, 0, y0=y0,t_bound= T) # initialising the Runge-Kutta solver

while solver.status == 'running': # needed because apparently this solver doesn't run itself
    solver.step()

    # Store the results at each time step
    times.append(solver.t)
    S,I = solver.y
    S_vals.append(S)
    I_vals.append(I)

plt.plot(times, S_vals, label='Susceptible')
plt.plot(times, I_vals, label='Infected', color='red')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIS Model Using RK45')
plt.show()