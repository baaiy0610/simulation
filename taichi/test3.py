import taichi as ti

ti.init(arch=ti.cpu)



#init mesh
init_x, init_y = 0.1, 0.6
N_x, N_y = 20, 4
N = N_x * N_y
dx = 1/32
N_edges = (N_x - 1) * N_y + (N_y - 1) * N_x + (N_x - 1) * (N_y - 1)


# physical quantities
m = 1
g = 9.8
YoungsModulus = 3e4

h = 16.7e-3
substepping = 100
dh = h/substepping

x = ti.Vector.field(2, ti.f32, N)
v = ti.Vector.field(2, ti.f32, N)
grad = ti.Vector.field(2, ti.f32, N)
spring_length = ti.field(ti.f32, N_edges)

edges = ti.Vector.field(2, ti.i32, N_edges)

@ti.func
def ij_2_index(i,j):
    return i*N_y + j

#meshing
@ti.kernel
def meshing():
    #horizon
    eid_base = 0
    for i in range(N_x-1):
        for j in range(N_y):
            eid = eid_base + i * N_y + j
            edges[eid] = [ij_2_index(i,j), ij_2_index(i+1,j)]
    eid_base += (N_x-1)*N_y
    #vertical
    for i in range(N_x):
        for j in range(N_y-1):
            eid = eid_base + i*(N_y-1) + j
            edges[eid] = [ij_2_index(i,j), ij_2_index(i,j+1)]
    #diagonal
    eid_base += (N_y-1) * N_x
    for i in range(N_x-1):
        for j in range(N_y-1):
            eid = eid_base + i*(N_y - 1) + j
            edges[eid] = [ij_2_index(i+1, j), ij_2_index(i,j+1)]

@ti.kernel
def initialize():
    for i in range(N_x):
        for j in range(N_y):
            index = ij_2_index(i,j)
            x[index] = ti.Vector([init_x + i*dx, init_y + j*dx])
            v[index] = ti.Vector([0.0 ,0.0])

# @ti.kernel
# def initialize():
#     for i, j in ti.ndarray(N_x, N_y):
#         index = ij_2_index(i,j)
#         x[index] = ti.Vector([init_x + i*dx, init_y + j*dx])
#         v[index] = ti.Vector([0.0 ,0.0])

@ti.kernel
def init_spring():
    # init spring rest-length
    for i in range(N_edges):
        a, b = edges[i][0], edges[i][1]
        r = x[a]-x[b]
        spring_length[i] = r.norm()

@ti.kernel
def compute_gridient():
    for i in grad:
        grad[i] = ti.Vector([0,0])
    
    for i in range(N_edges):
        a, b = edges[i][0], edges[i][1]
        r = x[a] - x[b]
        l = r.norm()
        l0 = spring_length[i]
        k = YoungsModulus/l0
        gradient = k*(l-l0)*r/l
        grad[a] += gradient
        grad[b] -= gradient

@ti.kernel
def update():
    for i in range(N):
        acc = -grad[i]/m - ti.Vector([0, g])
        v[i] += acc*dh
        x[i] += dh*v[i]
    
    for i in v:
        v[i] *= ti.exp(-dh*5)
    
    for j in range(N_y):
        ind = ij_2_index(0, j)
        v[ind] = ti.Vector([0,0])
        x[ind] = ti.Vector([init_x, init_y + j*dx])
    
    for i in range(N):
        if x[i][0]<init_x:
            x[i][0] = init_x
            v[i][0] = 0

meshing()
initialize()
init_spring()


gui = ti.GUI('mass-spring system', (800,800))
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.EXIT, ti.GUI.EXIT]:
            exit()
        elif e.key == "r":
            initialize()
    for i in range(substepping):
        compute_gridient()
        update()

    #render
    pos = x.to_numpy()
    for i in range(N_edges):
        a, b = edges[i][0], edges[i][1]
        gui.line((pos[a][0], pos[a][1]),
                 (pos[b][0], pos[b][1]),
                 radius=1,
                 color=0xFFFF00)
    gui.line((init_x, 0.0), (init_x, 1.0), color=0xFFFFFF, radius=4)
    gui.show()