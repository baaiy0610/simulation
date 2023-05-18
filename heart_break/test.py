import numpy as np
import matplotlib.pyplot as plt  
H=0.1
class Particle2D:
    def __init__(self, position : np.array) -> None:
        self.mass = 1
        self.position = position
        self.rho = np.zeros(position.shape[0])
        self.pressure = np.zeros_like(self.rho)
        self.p = np.zeros_like(self.rho)

    def draw_particles(self):  
        for i, (x, y) in enumerate(particles.position):  
            circ = plt.Circle((x, y), H / 2, color='r', alpha=0.3)  
            ax.add_artist(circ)  
            ax.set_aspect("equal")  
            ax.set_xlim([0, 10])  
            ax.set_ylim([0, 10])  

def initialize_particles(num_particles, fluid_domain):
    min_x, max_x = fluid_domain[0]
    min_y, max_y = fluid_domain[1]
    x = np.random.uniform(min_x, max_x, num_particles)
    y = np.random.uniform(min_y, max_y, num_particles)
    position = np.stack((x, y), axis=-1)
    return position

def update_particles(particles, dt, boundary):
    for particle in particles:
        particle.velocity += particle.acceleration * dt
        particle.position += particle.velocity * dt

        # Boundary handling
        if particle.position[0] < boundary[0][0]:
            particle.position[0] = 2 * boundary[0][0] - particle.position[0]
            particle.velocity[0] *= -1
        elif particle.position[0] > boundary[1][0]:
            particle.position[0] = 2 * boundary[1][0] - particle.position[0]
            particle.velocity[0] *= -1

        if particle.position[1] < boundary[0][1]:
            particle.position[1] = 2 * boundary[0][1] - particle.position[1]
            particle.velocity[1] *= -1
        elif particle.position[1] > boundary[1][1]:
            particle.position[1] = 2 * boundary[1][1] - particle.position[1]
            particle.velocity[1] *= -1
    
if __name__ == "__main__":
    num_particles = 1000
    fluid_domain = [(0,3),(0,3)]
    boudary = [(0,10),(0,10)]
    position = initialize_particles(num_particles, fluid_domain)
    particles = Particle2D(position)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))  
    fig.set_tight_layout(True)  
    particles.draw_particles()  
    plt.show()  
    plt.close()