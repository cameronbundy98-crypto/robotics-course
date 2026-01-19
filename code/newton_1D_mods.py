
import numpy as np
import matplotlib.pyplot as plt
import math

# Define the objective function and its gradient and Hessian
def f2(x, y):
    return (1-x)*(1-x)+10*(y-x*x)*(y-x*x)

def grad_f2(x, y):
    """Gradient of f"""
    grad_x = -2 * (1 - x) - 40*(y-x*x)*x 
	# 2 x -40 x y + 40 x^3  - 2
    grad_y = 20 * (y - x*x)
    return np.array([grad_x, grad_y])

def hess_f2(x, y):
    """Hessian of f"""
    h11 = 2-40*(y-x*x)+80*x*x
    h22 = 20
    h12 = -40*x
    return np.array([[h11, h12],
                     [h12, h22]])

# Define the objective function and its gradient and Hessian
def f(x):
    return 1/2*x*x-math.sin(x)

def grad_f(x):
    """Gradient of f"""
    return x-math.cos(x)

def hess_f(x):
    """Hessian of f"""
    return 1+math.sin(x)

def f(x):
    return math.sin(x)

def grad_f(x):
    """Gradient of f"""
    return math.cos(x)

def hess_f(x):
    """Hessian of f"""
    return -math.sin(x)

# Newton's Method
def newtons_method(start, tol=1e-6, max_iter=50):
    x = np.array(start, dtype=float)
    iterates = [x.copy()]
    optimal_values=[f(x)]
    
    for _ in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        while hess < 0.0:
            hess+=0.5
        
        # Newton step: x_new = x - H_inv * grad
        step = grad/hess
        x -= step
        
        iterates.append(x.copy())
        optimal_values.append(f(x))
        
        # Convergence check
        if np.linalg.norm(step) < tol:
            break
    
    return x, iterates, optimal_values


# Newton's Method
def newtons_method_BTLS(start, tol=1e-6, max_iter=50):
    x = np.array(start, dtype=float)
    iterates = [x.copy()]
    optimal_values=[f(x)]
    
    for _ in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        
        # Newton step: x_new = x - H_inv * grad
        step = grad/hess

        alpha=1.0
        while (f(x-alpha*step) > f(x) - 0.001*alpha*(grad * step ) ):
            alpha=alpha*0.6

        x -= alpha*step
        
        iterates.append(x.copy())
        optimal_values.append(f(x))
        
        # Convergence check
        if np.linalg.norm(step) < tol:
            break
    
    return x, iterates, optimal_values

plt.figure(figsize=(8, 6))
points = 1000 #Number of points
xmin, xmax = 0.0, 8.5
xlist=np.zeros(points)
ylist=np.zeros(points)
for i in range(0,points):
    xlist[i] = xmin+ (xmax-xmin)*i/points
    ylist[i] = f(xlist[i])

plt.plot(xlist,ylist,label="f(x) = sin(x)")



start_point = 5.8
# Run Newton's method
optimum, iterates, optimal_values_nm = newtons_method(start_point)
# Extract iterate points for plotting
iterates = np.array(iterates)
plt.plot(iterates, optimal_values_nm, 'o--', color="green",label=f"start={start_point}, iters = {iterates.size}")


start_point = 5.8
# Run Newton's method
optimum, iterates, optimal_values_nm_btls = newtons_method_BTLS(start_point)
# Extract iterate points for plotting
iterates = np.array(iterates)
plt.plot(iterates, optimal_values_nm_btls, 'o-', color="green",label=f"start={start_point}, iters = {iterates.size}")
# start_point = 0.5
# # Run Newton's method
# optimum, iterates, optimal_values_nm = newtons_method(start_point)
# # Extract iterate points for plotting
# iterates = np.array(iterates)
# plt.plot(iterates, optimal_values_nm, 'o--', color="red",label=f"start={start_point}, iters = {iterates.size}")


# start_point = 0.5
# # Run Newton's method
# optimum, iterates, optimal_values_nm_btls = newtons_method_BTLS(start_point)
# # Extract iterate points for plotting
# iterates = np.array(iterates)
# plt.plot(iterates, optimal_values_nm_btls, 'o-', color="red",label=f"start={start_point}, iters = {iterates.size}")



start_point = 5.9
# Run Newton's method
optimum, iterates, optimal_values_nm = newtons_method(start_point)
# Extract iterate points for plotting
iterates = np.array(iterates)
plt.plot(iterates, optimal_values_nm, 'o--', color="blue",label=f"start={start_point}, iters = {iterates.size}")


start_point = 5.9
# Run Newton's method
optimum, iterates, optimal_values_nm_btls = newtons_method_BTLS(start_point)
# Extract iterate points for plotting
iterates = np.array(iterates)
plt.plot(iterates, optimal_values_nm_btls, 'o-', color="blue",label=f"start={start_point}, iters = {iterates.size}")

start_point = 6.0
# Run Newton's method
optimum, iterates, optimal_values_nm = newtons_method(start_point)
# Extract iterate points for plotting
iterates = np.array(iterates)
plt.plot(iterates, optimal_values_nm, 'o--', color="purple",label=f"start={start_point}, iters = {iterates.size}")
start_point = 6.0
# Run Newton's method
optimum, iterates, optimal_values_nm_btls = newtons_method_BTLS(start_point)
# Extract iterate points for plotting
iterates = np.array(iterates)
plt.plot(iterates, optimal_values_nm_btls, 'o-', color="purple",label=f"start={start_point}, iters = {iterates.size}")




plt.xlabel("x_k")
plt.ylabel("f(x_k)")
plt.title("Modified Newton's Method: PD (dashed) BTLS (solid)")
plt.grid(True)
plt.legend()
plt.show()




