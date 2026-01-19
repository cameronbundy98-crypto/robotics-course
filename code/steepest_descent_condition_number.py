

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# Define the objective function and its gradient and Hessian

def f2(x, y, rho=1.0):
    return (x)**2+rho*(y)**2

def grad_f2(x, y,rho=1.0):
    """Gradient of f"""
    grad_x = 2 * x
    grad_y = 2 * rho*y
    return np.array([grad_x, grad_y])

def hess_f2(x, y,rho=1.0):
    """Hessian of f"""
    return np.array([[2.0, 0.0],
                     [0.0, 2.0*rho]])
# Steepest Descent
def steepest_descent(start, rho = 1.0,tol=1e-6, max_iter=5000):
    x = np.array(start, dtype=float)
    iterates = [x.copy()]
    optimal_values=[f2(x[0],x[1],rho)]
    for _ in range(max_iter):
        grad = grad_f2(x[0], x[1],rho)
        hess = hess_f2(x[0], x[1],rho)
        
        # Newton step: x_new = x - H_inv * grad
        alpha = (grad.T @ grad) / (grad.T @ hess @ grad)
        x -= alpha*grad
        
        iterates.append(x.copy())
        optimal_values.append(f2(x[0],x[1],rho))
        
        # Convergence check
        if np.linalg.norm(grad) < tol:
            break
    
    return x, iterates, optimal_values


# Plotting
x = np.linspace(-2, 2, 100)
y = np.linspace(-0.5, 1.5, 100)
X, Y = np.meshgrid(x, y)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6));
# Contour plot of the objective function
Z = f2(X, Y)
contour = ax1.contour(X, Y, Z, levels=5, cmap="Blues")
Z = f2(X, Y,rho=10.0)
contour = ax1.contour(X, Y, Z, levels=5, cmap="Purples")
Z = f2(X, Y,rho=100.0)
contour = ax1.contour(X, Y, Z, levels=5, cmap="Reds")
Z = f2(X, Y,rho=1000.0)
contour = ax1.contour(X, Y, Z, levels=5, cmap="Greens")

# Plot the iterates
ax1.plot(1, 1, 'x', color="blue", markersize=10, label="True Optimum (1, 1)")


#######################################
# Run Newton's method 
start_point = [1.5, 1.0]
# optimum, iterates, optimal_values_nm  = newtons_method(start_point)
# Extract iterate points for plotting
# iterates = np.array(iterates)
# x_iterates_nm, y_iterates_nm = iterates[:, 0], iterates[:, 1]
# ax1.plot(x_iterates_nm, y_iterates_nm, 'o-', color="green", label="NM Iterates")

#######################################
# Steepest Descent 
optimum, iterates, optimal_values_pt1 = steepest_descent([1.0,1/sqrt(2.0)],rho=2.0)
# Extract iterate points for plotting
iterates = np.array(iterates)
x_iterates, y_iterates = iterates[:, 0], iterates[:, 1]
ax1.plot(x_iterates, y_iterates, 'o-', color="blue", label="SD Iterates kappa=2.0")


#######################################
# Steepest Descent 
optimum, iterates, optimal_values_pt5 = steepest_descent([1.0,1/sqrt(10.0)],rho=10.0)
# Extract iterate points for plotting
iterates = np.array(iterates)
x_iterates, y_iterates = iterates[:, 0], iterates[:, 1]
ax1.plot(x_iterates, y_iterates, 'o-', color="purple", label="SD Iterates kappa=100.0")

#######################################
# Steepest Descent 
optimum, iterates, optimal_values_1 = steepest_descent([1.0,1/sqrt(100.0)],rho=100.0)
# Extract iterate points for plotting
iterates = np.array(iterates)
x_iterates, y_iterates = iterates[:, 0], iterates[:, 1]
ax1.plot(x_iterates, y_iterates, 'o-', color="red", label="SD Iterates alpha=0.01")


#######################################
# Steepest Descent 
optimum, iterates, optimal_values_2 = steepest_descent([1.0,1/sqrt(1000.0)],rho=1000.0)
# Extract iterate points for plotting
iterates = np.array(iterates)
x_iterates, y_iterates = iterates[:, 0], iterates[:, 1]
ax1.plot(x_iterates, y_iterates, 'o-', color="green", label="SD Iterates alpha=0.01")


# Labels and legend
ax1.set_xlabel("x_1")
ax1.set_ylabel("x_2")
ax1.set_title("Steepest Descent Iterates")


ax2.plot(range(0,len(optimal_values_pt1)),optimal_values_pt1,  'o-', color="blue", label="SD kappa = 2.0")
ax2.plot(range(0,len(optimal_values_pt5)),optimal_values_pt5,  'o-', color="purple", label="SD kappa = 10.0")
ax2.plot(range(0,len(optimal_values_1)),optimal_values_1,  'o-', color="red", label="SD kappa = 100.0")
ax2.plot(range(0,len(optimal_values_2)),optimal_values_2,  'o-', color="green", label="SD kappa = 1000.0")
# ax2.plot(range(0,len(optimal_values_nm)),optimal_values_nm,  'o-', color="green", label="Newton Method")
plt.xlabel("iteration number")
plt.ylabel("optimum")
ax2.set_title("$log f(x_k)$")
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()




