# import taichi as ti

# ti.init(arch=ti.gpu)
# n = 640
# pixels = ti.field(dtype=float, shape=(n, n))

# @ti.func
# def complex_sqr(z):
#     return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])

# @ti.kernel
# def paint(t: float):
#     for i, j in pixels:
#         # c = ti.Vector([-0.75, ti.cos(t)*0.11])
#         c = ti.Vector([ti.cos(t), ti.sin(t)])*0.7885
#         z = ti.Vector([i/n - 0.5, j/n - 0.5]) * 2
#         iterations = 0
#         while z.norm() < 10 and iterations < 100:
#             z = complex_sqr(z) + c
#             iterations += 1
#         pixels[i,j] = 1 - iterations * 0.02

# gui = ti.GUI("Julia Set", res=(n, n))

# for i in range(1000000):
#     paint(i*0.01)
#     gui.set_image(pixels)
#     gui.show()

import taichi as ti
ti.init(arch=ti.gpu)

n = 640
pixels = ti.field(dtype=float, shape=(n,n))

@ti.func
def complex_cub(z):
    return ti.Vector([z[0]**3 - z[1]**3 - 3*z[0]*z[1]**2, 3*z[0]**2 * z[1]])

@ti.kernel
def paint(t:float):
    for i, j in pixels:
        c = ti.Vector([-1*ti.cos(t), 0])
        z = ti.Vector([i/n - 0.5, j/n - 0.5]) * 2
        iteration = 0
        while z.norm()<10 and iteration < 100:
            z = complex_cub(z) + c
            iteration += 1
        pixels[i,j] = 1 - iteration * 0.02
    
gui = ti.GUI("Julia Set", res =(n,n))

for i in range(1000000):
    paint(i*0.01)
    gui.set_image(pixels)
    gui.show()