#!/usr/bin/env python
# coding: utf-8

# In[13]:


#Normal SIR

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Agent class for random walk and SIR states
class Agent:
    def __init__(self, status, position, R_circle):
        self.status = status  # 'S', 'I', or 'R'
        self.position = position  # Initial (x, y) position
        self.R_circle = R_circle  # Radius of the circular city
        self.path = [position.copy()]  # Track the path of the agent

    def random_walk(self, sigma):
        # Perform random walk with Gaussian displacement
        displacement = np.random.normal(0, sigma, 2)
        self.position += displacement

        # Boundary check: Teleport back inside if outside the circle
        if np.linalg.norm(self.position) > self.R_circle:
            theta = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, self.R_circle)
            self.position = np.array([radius * np.cos(theta), radius * np.sin(theta)])

        self.path.append(self.position.copy())  # Save current position

# Function to calculate distance between two agents
def distance(agent1, agent2):
    return np.linalg.norm(agent1.position - agent2.position)

# SIR agent-based model with random walk and infection spread
def update_sir(agents, infection_radius, alpha, beta):
    new_infected = []
    new_recovered = []

    for agent in agents:
        if agent.status == 'S':
            # Susceptible agent: Check proximity to infected agents
            for other_agent in agents:
                if other_agent.status == 'I' and distance(agent, other_agent) < infection_radius:
                    if np.random.rand() < alpha:  # Probability of infection
                        new_infected.append(agent)
                        break
        elif agent.status == 'I':
            # Infected agent: Check for recovery
            if np.random.rand() < beta:  # Probability of recovery
                new_recovered.append(agent)

    # Update states of newly infected and recovered agents
    for agent in new_infected:
        agent.status = 'I'
    for agent in new_recovered:
        agent.status = 'R'

# Animation of Random Walk and SIR updates for agents
def animate_sir_random_walk(agents, R_circle, infection_radius, alpha, beta, sigma, T):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set up the boundary circle
    circle = plt.Circle((0, 0), R_circle, color='black', fill=False, linestyle='--')
    ax.add_artist(circle)

    # Set axis limits
    ax.set_xlim(-R_circle, R_circle)
    ax.set_ylim(-R_circle, R_circle)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Random Walk and SIR Simulation')

    # Initialize scatter plot with agent positions and colors
    scat = ax.scatter([agent.position[0] for agent in agents], 
                      [agent.position[1] for agent in agents],
                      c=['blue' if agent.status == 'S' else 'red' if agent.status == 'I' else 'green' for agent in agents])

    # Create legend for Susceptible, Infected, and Recovered
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Susceptible')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Infected')
    green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Recovered')
    ax.legend(handles=[blue_patch, red_patch, green_patch])

    # Iteration counter to stop when no infected agents remain
    iteration_counter = [0]
    iteration_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')

    # SIR tracking for plot later
    susceptible, infected, recovered = [], [], []

    def update(frame):
        # Perform random walk and update SIR status at each step
        for agent in agents:
            agent.random_walk(sigma)

        # Update SIR status (infection spread and recovery)
        update_sir(agents, infection_radius, alpha, beta)

        # Update the scatter plot data with the new positions and states
        colors = ['blue' if agent.status == 'S' else 'red' if agent.status == 'I' else 'green' for agent in agents]
        scat.set_offsets([agent.position for agent in agents])
        scat.set_color(colors)

        # Increase the iteration counter
        iteration_counter[0] += 1
        iteration_text.set_text(f"Iteration: {iteration_counter[0]}")

        # Count the SIR agents for this iteration
        susceptible_count = sum(agent.status == 'S' for agent in agents)
        infected_count = sum(agent.status == 'I' for agent in agents)
        recovered_count = sum(agent.status == 'R' for agent in agents)

        susceptible.append(susceptible_count)
        infected.append(infected_count)
        recovered.append(recovered_count)

        # Check if all infected agents have recovered or remained susceptible
        if infected_count == 0:
            print(f"Simulation stopped after {iteration_counter[0]} iterations.")
            anim.event_source.stop()  # Stop the animation when no more infected agents
            plot_sir_counts(susceptible, infected, recovered, iteration_counter[0])  # Plot SIR data after simulation ends

        return scat, iteration_text

    anim = FuncAnimation(fig, update, frames=T, interval=50, repeat=False)
    plt.show()

# Plot SIR counts over time
def plot_sir_counts(susceptible, infected, recovered, iterations):
    plt.figure(figsize=(8, 6))
    t = np.arange(0, iterations)
    plt.plot(t, susceptible, 'b', label='Susceptible')
    plt.plot(t, infected, 'r', label='Infected')
    plt.plot(t, recovered, 'g', label='Recovered')
    plt.xlabel('Iterations')
    plt.ylabel('Population Count')
    plt.title('SIR Model over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Simulation parameters
N = 200  # Number of agents
S0 = 190  # Initial susceptible agents
I0 = 10   # Initial infected agents
R0 = 0    # Initial recovered agents

# Model parameters
R_circle = 2            # Circle radius
infection_radius = 0.1   # Infection radius
sigma = 0.03             # Standard deviation for random walk
alpha = 0.5              # Probability of infection upon contact
beta = 0.007             # Probability of recovery
T = 1500  # Maximum number of steps for animation and simulation

# Initialize agents
agents = []
for _ in range(S0):
    position = np.random.uniform(-R_circle, R_circle, 2)
    agents.append(Agent('S', position, R_circle))
for _ in range(I0):
    position = np.random.uniform(-R_circle, R_circle, 2)
    agents.append(Agent('I', position, R_circle))
for _ in range(R0):
    position = np.random.uniform(-R_circle, R_circle, 2)
    agents.append(Agent('R', position, R_circle))

# Run the random walk and SIR simulation animation
animate_sir_random_walk(agents, R_circle, infection_radius, alpha, beta, sigma, T)


# In[9]:


#Stochastic-SIR-Model

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Agent class for random walk and SEIRV states
class Agent:
    def __init__(self, status, position, R_circle):
        self.status = status  # 'S', 'E', 'I', 'R', or 'V'
        self.position = position  # Initial (x, y) position
        self.R_circle = R_circle  # Radius of the circular city
        self.path = [position.copy()]  # Track the path of the agent

    def random_walk(self, sigma):
        # Perform random walk with Gaussian displacement
        displacement = np.random.normal(0, sigma, 2)
        self.position += displacement

        # Boundary check: Teleport back inside if outside the circle
        if np.linalg.norm(self.position) > self.R_circle:
            theta = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, self.R_circle)
            self.position = np.array([radius * np.cos(theta), radius * np.sin(theta)])

        self.path.append(self.position.copy())  # Save current position

# Function to calculate distance between two agents
def distance(agent1, agent2):
    return np.linalg.norm(agent1.position - agent2.position)

# SEIRV agent-based model with random walk, exposure, infection spread, recovery, and vaccination
def update_seirv(agents, infection_radius, alpha, beta, delta, vaccination_rate):
    new_exposed = []
    new_infected = []
    new_recovered = []
    new_vaccinated = []

    for agent in agents:
        if agent.status == 'S':
            # Susceptible agent: Check proximity to infected agents
            for other_agent in agents:
                if other_agent.status == 'I' and distance(agent, other_agent) < infection_radius:
                    if np.random.rand() < alpha:  # Probability of exposure
                        new_exposed.append(agent)
                        break
            # Chance to vaccinate a susceptible agent
            if np.random.rand() < vaccination_rate:
                new_vaccinated.append(agent)
                
        elif agent.status == 'E':
            # Exposed agent: Transition to infected with probability delta
            if np.random.rand() < delta:
                new_infected.append(agent)
                
        elif agent.status == 'I':
            # Infected agent: Recover with probability beta
            if np.random.rand() < beta:
                new_recovered.append(agent)

    # Update states of newly exposed, infected, recovered, and vaccinated agents
    for agent in new_exposed:
        agent.status = 'E'
    for agent in new_infected:
        agent.status = 'I'
    for agent in new_recovered:
        agent.status = 'R'
    for agent in new_vaccinated:
        agent.status = 'V'

# Animation of Random Walk and SEIRV updates for agents
def animate_seirv_random_walk(agents, R_circle, infection_radius, alpha, beta, delta, vaccination_rate, T):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set up the boundary circle
    circle = plt.Circle((0, 0), R_circle, color='black', fill=False, linestyle='--')
    ax.add_artist(circle)

    # Set axis limits
    ax.set_xlim(-R_circle, R_circle)
    ax.set_ylim(-R_circle, R_circle)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Random Walk and SEIRV Simulation')

    # Initialize scatter plot with agent positions and colors
    scat = ax.scatter([agent.position[0] for agent in agents], 
                      [agent.position[1] for agent in agents],
                      c=['green' if agent.status == 'S' else 'yellow' if agent.status == 'E' else 'red' if agent.status == 'I' else 'blue' if agent.status == 'R' else 'purple' for agent in agents])

    # Create legend for the SEIRV model
    green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Susceptible')
    yellow_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Exposed')
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Infected')
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Recovered')
    purple_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Vaccinated')
    ax.legend(handles=[green_patch, yellow_patch, red_patch, blue_patch, purple_patch], loc='upper right')

    # Start the timer
    start_time = time.time()

    # Iteration counter to stop when no infected agents remain
    iteration_counter = [0]
    iteration_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')

    # SEIRV tracking for plot later
    susceptible, exposed, infected, recovered, vaccinated = [], [], [], [], []

    def update(frame):
        # Perform random walk and update SEIRV status at each step
        for agent in agents:
            agent.random_walk(sigma)

        # Update SEIRV status (exposure, infection spread, recovery, and vaccination)
        update_seirv(agents, infection_radius, alpha, beta, delta, vaccination_rate)

        # Update the scatter plot data with the new positions and states
        colors = ['green' if agent.status == 'S' else 'yellow' if agent.status == 'E' else 'red' if agent.status == 'I' else 'blue' if agent.status == 'R' else 'purple' for agent in agents]
        scat.set_offsets([agent.position for agent in agents])
        scat.set_color(colors)

        # Increase the iteration counter
        iteration_counter[0] += 1
        iteration_text.set_text(f"Iteration: {iteration_counter[0]}")

        # Count the SEIRV agents for this iteration
        susceptible_count = sum(agent.status == 'S' for agent in agents)
        exposed_count = sum(agent.status == 'E' for agent in agents)
        infected_count = sum(agent.status == 'I' for agent in agents)
        recovered_count = sum(agent.status == 'R' for agent in agents)
        vaccinated_count = sum(agent.status == 'V' for agent in agents)

        susceptible.append(susceptible_count)
        exposed.append(exposed_count)
        infected.append(infected_count)
        recovered.append(recovered_count)
        vaccinated.append(vaccinated_count)

        # Check if all infected agents have recovered, remained susceptible, or vaccinated
        if infected_count == 0:
            end_time = time.time()  # End the timer
            elapsed_time = end_time - start_time  # Calculate elapsed time
            mins, secs = divmod(elapsed_time, 60)
            print(f"Simulation stopped after {int(mins)} minutes and {int(secs)} seconds.")
            anim.event_source.stop()  # Stop the animation when no more infected agents
            plot_seirv_counts(susceptible, exposed, infected, recovered, vaccinated, iteration_counter[0])  # Plot SEIRV data after simulation ends

        return scat, iteration_text

    anim = FuncAnimation(fig, update, frames=T, interval=50, repeat=False)
    plt.show()

# Plot SEIRV counts over time
def plot_seirv_counts(susceptible, exposed, infected, recovered, vaccinated, iterations):
    plt.figure(figsize=(8, 6))
    t = np.arange(0, iterations)
    plt.plot(t, susceptible, 'green', label='Susceptible')
    plt.plot(t, exposed, 'yellow', label='Exposed')
    plt.plot(t, infected, 'red', label='Infected')
    plt.plot(t, recovered, 'blue', label='Recovered')
    plt.plot(t, vaccinated, 'purple', label='Vaccinated')
    plt.xlabel('Iterations')
    plt.ylabel('Population Count')
    plt.title('SEIRV Model over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Simulation parameters
N = 200  # Number of agents
S0 = 190  # Initial susceptible agents
E0 = 0    # Initial exposed agents
I0 = 10   # Initial infected agents
R0 = 0    # Initial recovered agents
V0 = 0    # Initial vaccinated agents

# Model parameters
R_circle = 2            # Circle radius
infection_radius = 0.1   # Infection radius
sigma = 0.03             # Standard deviation for random walk
alpha = 0.5              # Probability of exposure upon contact
beta = 0.007             # Probability of recovery
delta = 0.1              # Probability of becoming infectious after exposure
vaccination_rate = 0.002  # Probability of vaccination for susceptible agents
T = 1000  # Maximum number of steps for animation and simulation

# Initialize agents
agents = []
for _ in range(S0):
    position = np.random.uniform(-R_circle, R_circle, 2)
    agents.append(Agent('S', position, R_circle))
for _ in range(E0):
    position = np.random.uniform(-R_circle, R_circle, 2)
    agents.append(Agent('E', position, R_circle))
for _ in range(I0):
    position = np.random.uniform(-R_circle, R_circle, 2)
    agents.append(Agent('I', position, R_circle))
for _ in range(R0):
    position = np.random.uniform(-R_circle, R_circle, 2)
    agents.append(Agent('R', position, R_circle))
for _ in range(V0):
    position = np.random.uniform(-R_circle, R_circle, 2)
    agents.append(Agent('V', position, R_circle))

# Run the random walk and SEIRV simulation animation
animate_seirv_random_walk(agents, R_circle, infection_radius, alpha, beta, delta, vaccination_rate, T)


# In[ ]:


# Stochastic model of SIR

import numpy as np
import matplotlib.pyplot as plt

# Model parameters 
alpha = 0.05   # Probability of infection
beta = 0.01    # Probability of recovery
gamma = 0.001  # Probability of losing immunity (becoming susceptible again)

# Population size and simulation parameters
N = 100        # Number of agents (people)
T = 500        # Number of time steps

# Transition probability matrix 
transition_matrix = np.array([
    [1 - alpha, alpha, 0],  # From Susceptible
    [0, 1 - beta, beta],    # From Infected
    [gamma, 0, 1 - gamma]   # From Recovered
])

# Initial states for each agent (99% Susceptible, 1% Infected, 0% Recovered)
initial_states = np.random.choice([0, 1, 2], size=N, p=[0.99, 0.01, 0.0])  # 0=S, 1=I, 2=R

# Arrays to store the proportions of S, I, R over time
S = np.zeros(T)
I = np.zeros(T)
R = np.zeros(T)

# Set initial proportions
S[0] = np.sum(initial_states == 0) / N
I[0] = np.sum(initial_states == 1) / N
R[0] = np.sum(initial_states == 2) / N

# Function to simulate the transition of one agent
def simulate_one_step(states, transition_matrix):
    new_states = np.zeros_like(states)
    for i in range(N):
        current_state = states[i]
        # Transition based on current state and transition matrix probabilities
        new_states[i] = np.random.choice([0, 1, 2], p=transition_matrix[current_state])
    return new_states

# Simulate the epidemic over T time steps
for t in range(1, T):
    initial_states = simulate_one_step(initial_states, transition_matrix)
    # Calculate proportions of S, I, R at each time step
    S[t] = np.sum(initial_states == 0) / N
    I[t] = np.sum(initial_states == 1) / N
    R[t] = np.sum(initial_states == 2) / N

# Plot the results: Proportions of Susceptible, Infected, and Recovered over time
plt.figure(figsize=(10, 6))
plt.plot(S, label='Susceptible', color='blue')
plt.plot(I, label='Infected', color='red')
plt.plot(R, label='Recovered', color='green')
plt.xlabel('Time Steps')
plt.ylabel('Proportion')
plt.title('Probabilistic Epidemic Simulation (Over Time)')
plt.legend()
plt.grid(True)
plt.show()

