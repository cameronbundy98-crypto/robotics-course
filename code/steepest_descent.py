
import numpy as np
import matplotlib.pyplot as plt

# Define the objective function and its gradient and Hessian

def f2(x, y):
    return (1-x)**2+10*(y-x*x)**2

def grad_f2(x, y):
    """Gradient of f"""
    grad_x = -2*(1-x)-40*(y-x*x)*x
    grad_y = 20 * (y - x*x)
    return np.array([grad_x, grad_y])

def hess_f(x, y):
    """Hessian of f"""
    h11 = 2 + 80*x*x-40*(y-x*x)
    h12 = -40*x
    h22 = 20
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

# Steepest Descent
def steepest_descent(start, alpha=0.00001,tol=1e-4, max_iter=20000):
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


#######################################
# Run Newton's method 
start_point = [0, 1.0]
# optimum, iterates, optimal_values_nm  = newtons_method(start_point)
# Extract iterate points for plotting
# iterates = np.array(iterates)
# x_iterates_nm, y_iterates_nm = iterates[:, 0], iterates[:, 1]
# ax1.plot(x_iterates_nm, y_iterates_nm, 'o-', color="green", label="NM Iterates")

#######################################
# Steepest Descent 
optimum, iterates, optimal_values_pt1 = steepest_descent(start_point,alpha=0.001)
# Extract iterate points for plotting
iterates = np.array(iterates)
x_iterates, y_iterates = iterates[:, 0], iterates[:, 1]
ax1.plot(x_iterates, y_iterates, 'o-', color="blue", label="SD Iterates alpha=0.001")


#######################################
# Steepest Descent 
optimum, iterates, optimal_values_pt5 = steepest_descent(start_point,alpha=0.005)
# Extract iterate points for plotting
iterates = np.array(iterates)
x_iterates, y_iterates = iterates[:, 0], iterates[:, 1]
ax1.plot(x_iterates, y_iterates, 'o-', color="purple", label="SD Iterates alpha=0.005")

#######################################
# Steepest Descent 
optimum, iterates, optimal_values_1 = steepest_descent(start_point,alpha=0.020)
# Extract iterate points for plotting
iterates = np.array(iterates)
x_iterates, y_iterates = iterates[:, 0], iterates[:, 1]
ax1.plot(x_iterates, y_iterates, 'o-', color="red", label="SD Iterates alpha=0.01")

# Annotate start and end points
ax1.annotate("Start", (x_iterates[0], y_iterates[0]), textcoords="offset points", xytext=(-10, 10), ha="center", color="red")
ax1.annotate("End", (x_iterates[-1], y_iterates[-1]), textcoords="offset points", xytext=(-10, -15), ha="center", color="red")

# Labels and legend
ax1.set_xlabel("x_1")
ax1.set_ylabel("x_2")
ax1.set_title("Steepest Descent Iterates")


ax2.plot(range(0,len(optimal_values_pt1)),optimal_values_pt1,  'o-', color="blue", label="SD alpha = 0.001")
ax2.plot(range(0,len(optimal_values_pt5)),optimal_values_pt5,  'o-', color="purple", label="SD alpha = 0.005")
ax2.plot(range(0,len(optimal_values_1)),optimal_values_1,  'o-', color="red", label="SD alpha = 0.01")
# ax2.plot(range(0,len(optimal_values_nm)),optimal_values_nm,  'o-', color="green", label="Newton Method")
plt.xlabel("iteration number")
plt.ylabel("optimum")
ax2.set_title("$log f(x_k)$")
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()




