
import numpy as np
import matplotlib.pyplot as plt

# Define the objective function and its gradient and Hessian
def f2(x, y):
    return (1-x)*(1-x)+10*(y-x*x)*(y-x*x)

def grad_f2(x, y):
    """Gradient of f"""
    grad_x = -2 * (1 - x) - 40*(y-x*x)*x 
	# 2 x -40 x y + 40 x^3  - 2
    grad_y = 20 * (y - x*x)
    return np.array([grad_x, grad_y])

def hess_f(x, y):
    """Hessian of f"""
    h11 = 2-40*(y-x*x)+80*x*x
    h22 = 20
    h12 = -40*x
    return np.array([[h11, h12],
                     [h12, h22]])

# Newton's Method
def newtons_method(start, tol=1e-6, max_iter=50):
    x = np.array(start, dtype=float)
    iterates = [x.copy()]
    optimal_values=[f2(x[0],x[1])]
    
    for _ in range(max_iter):
        grad = grad_f2(x[0], x[1])
        hess = hess_f(x[0], x[1])
        
        # Newton step: x_new = x - H_inv * grad
        step = np.linalg.solve(hess, grad)
        x -= step
        
        iterates.append(x.copy())
        optimal_values.append(f2(x[0],x[1]))
        
        # Convergence check
        if np.linalg.norm(step) < tol:
            break
    
    return x, iterates, optimal_values


# Newton's Method
def newtons_method_BTLS(start, tol=1e-6, max_iter=50):
    x = np.array(start, dtype=float)
    iterates = [x.copy()]
    optimal_values=[f2(x[0],x[1])]
    
    for _ in range(max_iter):
        grad = grad_f2(x[0], x[1])
        hess = hess_f(x[0], x[1])
        
        # Newton step: x_new = x - H_inv * grad
        step = np.linalg.solve(hess, grad)

        alpha=1.0
        while (f2(x[0]-alpha*step[0],x[1]-alpha*step[1]) > f2(x[0],x[1]) - 0.1*alpha*(np.linalg.norm(grad.T @ step )) ):
            alpha=alpha*0.5

        x -= alpha*step
        
        iterates.append(x.copy())
        optimal_values.append(f2(x[0],x[1]))
        
        # Convergence check
        if np.linalg.norm(step) < tol:
            break
    
    return x, iterates, optimal_values

# Steepest Descent
def steepest_descent(start, alpha=0.00001,tol=1e-6, max_iter=5000):
    x = np.array(start, dtype=float)
    iterates = [x.copy()]
    optimal_values=[f2(x[0],x[1])]
    for _ in range(max_iter):
        grad = grad_f2(x[0], x[1])
        
        # Newton step: x_new = x - H_inv * grad
        x -= alpha*grad
        
        iterates.append(x.copy())
        optimal_values.append(f2(x[0],x[1]))
        
        # Convergence check
        if np.linalg.norm(grad) < tol:
            break
    
    return x, iterates, optimal_values


# Steepest Descent
def steepest_descent_BTLS(start, tol=1e-6, max_iter=5000):
    x = np.array(start, dtype=float)
    iterates = [x.copy()]
    optimal_values=[f2(x[0],x[1])]
    for _ in range(max_iter):
        grad = grad_f2(x[0], x[1])
        
        alpha=1.0
        while (f2(x[0]-alpha*grad[0],x[1]-alpha*grad[1]) > f2(x[0],x[1]) - 0.1*alpha*(np.linalg.norm(grad))**2 ):
            alpha=alpha*0.4

        x -= alpha*grad
        
        iterates.append(x.copy())
        optimal_values.append(f2(x[0],x[1]))
        
        # Convergence check
        if np.linalg.norm(grad) < tol:
            break
    
    return x, iterates, optimal_values

start_point = [0.1, 0.2]
# Run Newton's method
optimum, iterates, optimal_values_nm = newtons_method(start_point)
# Extract iterate points for plotting
iterates = np.array(iterates)
x_iterates_nm, y_iterates_nm = iterates[:, 0], iterates[:, 1]


# Run Newton's method
optimum, iterates, optimal_values_nm_btls = newtons_method_BTLS(start_point)
# Extract iterate points for plotting
iterates = np.array(iterates)
x_iterates_nm_btls, y_iterates_nm_btls = iterates[:, 0], iterates[:, 1]

# Plotting
x = np.linspace(-2, 2, 100)
y = np.linspace(-0.5, 1.5, 100)
X, Y = np.meshgrid(x, y)
Z = f2(X, Y)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6));
# Contour plot of the objective function
contour = ax1.contour(X, Y, Z, levels=200, cmap="viridis")
ax1.clabel(contour, inline=True, fontsize=8)
plt.colorbar(contour, label="Objective Function Value")

# Plot the iterates
ax1.plot(1, 1, 'x', color="blue", markersize=10, label="True Optimum (1, 1)")


optimum, iterates, optimal_values_BTLS = steepest_descent_BTLS(start_point)
# Extract iterate points for plotting
iterates = np.array(iterates)
x_iterates, y_iterates = iterates[:, 0], iterates[:, 1]
ax1.plot(x_iterates, y_iterates, 'o-', color="green", label="SD BTLS Iterates")

ax1.plot(x_iterates_nm, y_iterates_nm, 'o-', color="black", label="NM Iterates")
ax1.plot(x_iterates_nm_btls, y_iterates_nm_btls, 'o-', color="cyan", label="NM Iterates")
# Annotate start and end points
ax1.annotate("Start", (x_iterates[0], y_iterates[0]), textcoords="offset points", xytext=(-10, 10), ha="center", color="red")
ax1.annotate("End", (x_iterates[-1], y_iterates[-1]), textcoords="offset points", xytext=(-10, -15), ha="center", color="red")

# Labels and legend
ax1.set_xlabel("x_1")
ax1.set_ylabel("x_2")
ax1.set_title("Newton Method Iterates")
#ax1.legend()

if True:
    ax2.plot(range(0,len(optimal_values_BTLS[0:30])),optimal_values_BTLS[0:30],  'o-', color="green", label="SD BTLS")
ax2.plot(range(0,len(optimal_values_nm)),optimal_values_nm,  'o-', color="black", label="NM")
ax2.plot(range(0,len(optimal_values_nm_btls)),optimal_values_nm_btls,  'o-', color="cyan", label="NM BTLS")
plt.xlabel("iteration number")
plt.ylabel("optimum")
ax2.set_title("$log f(x_k)$")
ax2.legend()
plt.yscale('log')
plt.show()




