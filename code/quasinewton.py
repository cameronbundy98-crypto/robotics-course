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


def phi(x,p,alpha):
    return f2(x[0]+alpha*p[0],x[1]+alpha*p[1])

def phip(x,p,alpha):
    return grad_f2(x[0]+alpha*p[0],x[1]+alpha*p[1]).T @ p

def interp(x,p,alpha_lo,alpha_hi):
    ## phi(alpha) = f(x_k + alpha p_k), alpha > 0
    phi_lo = f2(x[0]+alpha_lo*p[0],x[1]+alpha_lo*p[1]) ## phi(alpha_lo)
    phi_hi = f2(x[0]+alpha_hi*p[0],x[1]+alpha_hi*p[1]) ## phi(alpha_hi)
    new_grad = grad_f2(x[0]+alpha_lo*p[0],x[1]+alpha_lo*p[1])  ## grad(xk + alpha pk)
    phip_lo = new_grad.T @ p ## phi'(alpha_lo)
    #  print("alpha interp before correction", -( phip_lo*((alpha_hi-alpha_lo)**2) )/(2 *(phi_hi-phi_lo-phip_lo*(alpha_hi-alpha_lo)) ) ) 
    alpha_interp = alpha_lo-( phip_lo*((alpha_hi-alpha_lo)**2) )/(2 *(phi_hi-phi_lo-phip_lo*(alpha_hi-alpha_lo)) )
    if (alpha_interp <= alpha_lo+0.001 or alpha_interp >= alpha_hi-0.001):
        alpha_interp = (alpha_lo+alpha_hi)/2
        #  print("error in interp", alpha_interp)
    return alpha_interp


def zoom(x,p,alpha_lo,alpha_hi):
    c1 = 0.0001
    c2 = 0.5
    count = 0
    while count<100:
        alpha_j = interp(x,p,alpha_lo,alpha_hi)
        #  print(f"[{alpha_lo},{alpha_j},{alpha_hi}]")
        grad = grad_f2(x[0],x[1])
        flag = f2(x[0]+alpha_j*p[0],x[1]+alpha_j*p[1]) > f2(x[0],x[1]) + c1*alpha_j*(grad.T @ p) 
        flag2 = f2(x[0]+alpha_j*p[0],x[1]+alpha_j*p[1]) >= f2(x[0]+alpha_lo*p[0],x[1]+alpha_lo*p[1]) ## phi(alpha_j) >= phi(alpha_lo)
        if flag or flag2 : ## If Armijo condition fails at alpha_j, or we are increasing objective at alpha_j relative to alpha_lo, we have to reduce upper limit alpha_hi
            alpha_hi = alpha_j
        else:
            grad_new = grad_f2(x[0]+alpha_j*p[0],x[1]+alpha_j*p[1])
            phip_new = grad_new.T @ p ## phi'(alpha_j)
            if abs(phip_new) <=  -c2* (grad.T @ p): ## 
                return alpha_j
            
            if phip_new*(alpha_hi-alpha_lo) >= 0.0:
                alpha_hi = alpha_lo
            alpha_lo=alpha_j
        #  print(f"post flags in zoom: [{alpha_lo},{alpha_j},{alpha_hi}]")
        if abs(alpha_j)< 1e-6:
            #  print("error in zoom")
            break
        count+=1
        
    return alpha_j ## error!






def step_size_calc(x,p,alpha_max):
    c1 = 0.0001
    c2 = 0.5
    alpha_prev = 0
    alpha = alpha_max*2/3; 
    count = 0
    while count <100 :

        #  print(f"step size count {count}: [{alpha_prev},{alpha}]")
        flag = (f2(x[0]+alpha*p[0],x[1]+alpha*p[1]) > f2(x[0],x[1]) + c1*alpha*phip(x,p,0.0) )
        flag2 = phi(x,p,alpha) >= phi(x,p,alpha_prev) and count > 0
        if flag or flag2:
            #  print("flag1 and 2",flag,flag2)
            return zoom(x,p,alpha_prev,alpha)
        phip_new = phip(x,p,alpha)
        flag3 = abs(phip_new) <= -c2*phip(x,p,0)
        if flag3:
            #  print("flag3")
            return alpha
        if phip_new >= 0:
            #  print("\n\n\nREV zoom: ",zoom(x,p,alpha,alpha_prev))
            return zoom(x,p,alpha,alpha_prev)
        #  print("no flags triggered. Switching over to [alpha_i,alpha_max] interval")
        
        alpha_prev = alpha
        alpha = (alpha+alpha_max)/2
        #  print(f"new step size count {count}: [{alpha_prev},{alpha}]")
        count+=1
        if alpha <= 1e-6:
            #  print("error during step size calc")
            break

    return alpha ## error


# Newton's Method
def quasi_newton_method(start, tol=1e-6, max_iter=120):
    x = np.array(start, dtype=float)
    iterates = [x.copy()]
    optimal_values=[f2(x[0],x[1])]
    prev_grad = np.zeros(2)
    grad = grad_f2(x[0], x[1])
    Bk = np.array([[1.0,0.0],[0.0,1.0]]) * 1.0
    Hk = np.array([[1.0,0.0],[0.0,1.0]]) 
    Id = np.array([[1.0,0.0],[0.0,1.0]]) 
    
    for count in range(max_iter):
        #  print(f"\niter: {count}")
        # Newton step: x_new = x - H_inv * grad

        ## Propagating Bk: 
        # step = np.linalg.solve(Bk, grad)
        step = Hk @ grad
        alpha=1.0
        #  print(Hk)
        # Backtracking: 
        while (f2(x[0]-alpha*step[0],x[1]-alpha*step[1]) > f2(x[0],x[1]) - 0.0001*alpha*(grad.T @ step) ):
            alpha=alpha*0.5
        #  print("\nbacktrack: ",alpha)
        ## Strong wolfe:
        alpha = step_size_calc(x,-step,1.0)
        #  print(f"{count}: result of wolfe: ",alpha)

        x -= alpha*step ## xk -> x k+1
        iterates.append(x.copy())
        optimal_values.append(f2(x[0],x[1]))

        prev_grad = grad.copy() ## grad xk
        grad = grad_f2(x[0], x[1]) ## grad x k+1


        yk = grad-prev_grad
        sk = iterates[count+1]-iterates[count] 
        # if (yk.T @ sk) < 1e-4:
        #     Bk = np.array([[1.0,0.0],[0.0,1.0]]) * (grad.T @ grad) / 1.0
        # v = yk-hess @ sk
        #  # print("yk-Bk sk",v)
        # hess = hess + (v @ v.T ) / (v.T @ sk)

        # #  BFGS propagating Bk
        # Bk = Bk - ((Bk @ sk) @ (sk.T  @ Bk))/(sk.T @ Bk @ sk) + (yk @ yk.T) / (yk.T @ sk) ##


        # BFGS propagating Hk. worked better with backtracking, under heuristic initial Hk and resetting
        rho_k = 1/(sk.T @ yk)
        li = (Id - rho_k*(yk @ sk.T))
        Hk = li.T @ Hk @ li + rho_k *( sk @ sk.T) 
        # if (yk.T @ sk) < 1e-3:
        #  #     print("correcting Hk")
        #     Hk = np.array([[1.0,0.0],[0.0,1.0]]) * (grad.T @ grad) / 1.0

        # # Convergence check
        # if np.linalg.norm(step) < tol:
            # break
    
    return x, iterates, optimal_values


start_point = [0.0, 1.2]
# Run Newton's method
optimum, iterates, optimal_values_nm = newtons_method(start_point)
# Extract iterate points for plotting
iterates = np.array(iterates)
x_iterates_nm, y_iterates_nm = iterates[:, 0], iterates[:, 1]


# Run Newton's method
optimum, iterates, optimal_values_nm_btls = quasi_newton_method(start_point)
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



ax1.plot(x_iterates_nm, y_iterates_nm, 'o-', color="black", label="NM Iterates")
ax1.plot(x_iterates_nm_btls, y_iterates_nm_btls, 'o-', color="cyan", label="QNM Iterates")
# Annotate start and end points

# Labels and legend
ax1.set_xlabel("x_1")
ax1.set_ylabel("x_2")
ax1.set_title("Newton Method Iterates")
#ax1.legend()

ax2.plot(range(0,len(optimal_values_nm)),optimal_values_nm,  'o-', color="black", label="NM")
ax2.plot(range(0,len(optimal_values_nm_btls)),optimal_values_nm_btls,  'o-', color="cyan", label="QNM")
plt.xlabel("iteration number")
plt.ylabel("optimum")
ax2.set_title("$log f(x_k)$")
ax2.legend()
plt.yscale('log')
plt.show()




