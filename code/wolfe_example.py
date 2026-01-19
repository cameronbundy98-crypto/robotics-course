import numpy as np
import matplotlib.pyplot as plt

# Define the objective function and its gradient and Hessian
def f(x):
    return (x-3)**2

def grad_f(x):
    """Gradient of f"""
    return 2*(x-3)

def hess_f(x):
    """Hessian of f"""
    return 2.0

def myplot(xlist,ylist,func,x,grad):
    points = len(xlist)
    for i in range(0,points):
        ylist[i] = func(xlist[i])
    plt.plot(xlist, ylist,'-',color="black")
    for i in range(0,points):
        ylist[i] = func(x)+0.1*grad*((xlist[i]-x))
    plt.plot(xlist, ylist,color="green")
    for i in range(0,points):
        ylist[i] = func(x)+(xlist[i]-x)*grad
    plt.plot(xlist, ylist,color="blue",label=f"tangent at {x}")


def myplot_tangent(xlist,ylist,func,x,grad):
    points = len(xlist)
    for i in range(0,points):
        ylist[i] = func(x)+(xlist[i]-x)*grad
    plt.plot(xlist, ylist,color="green",linestyle="dashed",label=f"tangent at {x} whose slope is 0.5x slope of tangent at 5.5")

def steepest_descent_BTLS(start_point=5.5,tol=1e-6,max_iter=50,alpha0=1.0):
    x = start_point
    iterates = [x]
    optimal_values=[f(x)]

    points = 1000 #Number of points
    xmin, xmax = 0.0, 6.0
    xlist=np.zeros(points)
    ylist=np.zeros(points)
    for i in range(0,points):
        xlist[i] = xmin+ (xmax-xmin)*i/points
    grad = grad_f(x)
    step=-grad
    myplot(xlist,ylist,f,x,grad)
    myplot_tangent(xlist,ylist,f,4.25,grad_f(4.25))
    plt.plot(4.25,f(4.25),'.',markersize=10,color="orange",label="(4.25,f(4.25))")

    plt.plot(x,f(x),'.',markersize=10,color="blue")
    flag=f(x+alpha0*step) > f(x) + 0.1*alpha0*grad*step
    plt.plot(x+step*alpha0,f(x+step*alpha0),'s',markersize=10,color="red",label=f"alpha={alpha0} (too large)" )
    alpha0=0.7*alpha0
    plt.plot(x+step*alpha0,f(x+step*alpha0),'o',markersize=10,color="green",label=f"alpha={alpha0} (satisfies Wolfe conditions)")
    alpha0=0.05
    plt.plot(x+step*alpha0,f(x+step*alpha0),'d',markersize=10,color="purple",label=f"alpha ={alpha0} (too small)")
    plt.plot([1,4.25] ,[-1.0,-1.0], '|-', color="red", label="Acceptable range from Wolfe conditions")
    plt.ylabel("f(x)")
    plt.xlabel("x")
    plt.legend()
    plt.grid(True)
    plt.title("$c_1 = 0.1, c_2 = 0.5, alpha=1 , 0.1$")


plt.figure(figsize=(8, 6))
steepest_descent_BTLS()

plt.show()
