# Surfing problem

from dolfin import *
import numpy as np
import time


set_log_level(40)  # Error level=40, warning level=30
parameters["linear_algebra_backend"] = "PETSc"
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["quadrature_degree"] = 4


parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {
    "optimize": True,
    "eliminate_zeros": True,
    "precompute_basis_const": True,
    "precompute_ip_const": True,
}

comm = MPI.comm_world
comm_rank = MPI.rank(comm)

# Create mesh and define function space
# mesh=Mesh("/fenics/Ss_eps12_h_12e-3.xml")
mesh = Mesh("./mesh/surf_tri.xml")
h = FacetArea(mesh)  # area/length of a cell facet on a given mesh
h_avg = (h("+") + h("-")) / 2
n = FacetNormal(mesh)

# Choose phase-field model
phase_model = 1
# 1 for linear model, 2 for quadratic model

# Elasticity parameters
E, nu = 9800, 0.13
mu, lmbda, kappa = (
    Constant(E / (2 * (1 + nu))),
    Constant(E * nu / ((1 + nu) * (1 - 2 * nu))),
    Constant(E / (3 * (1 - 2 * nu))),
)
# Fracture parameters (NOTE: k is Gc, lch is l)
# k, lch, eta1, eta2 = 0.091125, 0.12, 0.0, 0.0
# k, lch, eta1, eta2 = 0.091125, 0.4, 0.0, 0.0
sts = 27
scs = 77
shs = (2 / 3) * sts * scs / (scs - sts)
Wts = sts**2 / (2 * E)
Whs = shs**2 / (2 * kappa)
eta1, eta2 = 0, 0

# beta3 = (lch * sts) / (mu * kappa * k)

# beta0 = -2.75
# beta1 = 0.0390723
# beta2 = 0.0705819


# pen = 180 * 160 * k / lch  # (-beta0) k/lch # NOTE: penalty coefficient

Gc = 0.091
lch = 3 * Gc * E / 8 / (sts**2)  # lch = 0.459375
eps = 0.35  # eps <= lch/4
h_size = 0.1
delta = (1 + 3 * h_size / (8 * eps)) ** (-2) * ((sts + (1 + 2 * np.sqrt(3)) * shs) / ((8 + 3 * np.sqrt(3)) * shs)) * 3 * Gc / (16 * Wts * eps) + (
    1 + 3 * h_size / (8 * eps)
) ** (-1) * (2 / 5)

pen = 1000 * (3 * Gc / 8 / eps) * conditional(lt(delta, 1), 1, delta)  # NOTE: 24 brizian

V = VectorFunctionSpace(mesh, "CG", 1)  # Function space for u
Y = FunctionSpace(mesh, "CG", 1)  # Function space for z

# Mark boundary subdomians
left = CompiledSubDomain("near(x[0], side, tol) && abs(x[1]) > tol && on_boundary", side=0.0, tol=1e-4)
right = CompiledSubDomain("near(x[0], side, tol) && on_boundary", side=30.0, tol=1e-4)  # 30.0
bottom = CompiledSubDomain("near(x[1], side, tol) && on_boundary", side=-5.0, tol=1e-4)  # 5.0
top = CompiledSubDomain("near(x[1], side, tol) && on_boundary", side=5.0, tol=1e-4)


def lefttop(x):
    # return abs(x[0]-30.0)<1e-4 and abs(x[1]-5.0)<1e-4
    return abs(x[0]) < 1e-4 and abs(x[1] + 5) < 1e-4  # leftbottom


def outer(x):
    return x[1] > 0.5


def corner(x):
    # return abs(x[1]-0)<1e-4 and x[0]<5+1e-4 and x[0]>4.5
    return abs(x[1]) < 1e-4 and x[0] < 5 + 1e-4


##################################################################################
# Define Dirichlet boundary (x = 0 or x = 1)
##################################################################################

# c = Expression(
#     "K1/(2*mu)*sqrt(sqrt(pow(x[0]-V*(t+0.1),2)+pow(x[1],2))/(2*pi))*(kap-cos(atan2(x[1],(x[0]-V*(t+0.1)))))*cos(atan2(x[1],(x[0]-V*(t+0.1)))/2)",
#     degree=4,
#     t=0,
#     V=20,
#     K1=30,
#     mu=4336.28,
#     kap=2.54,
# )
# r = Expression(
#     "K1/(2*mu)*sqrt(sqrt(pow(x[0]-V*(t+0.1),2)+pow(x[1],2))/(2*pi))*(kap-cos(atan2(x[1],(x[0]-V*(t+0.1)))))*sin(atan2(x[1],(x[0]-V*(t+0.1)))/2)",
#     degree=4,
#     t=0,
#     V=20,
#     K1=30,
#     mu=4336.28,
#     kap=2.54,
# )

c1 = (1 + nu) * sqrt(Gc) / sqrt(2 * pi * E)
c2 = (3 - nu) / (1 + nu)
ahead = 2
vel = 20

r = Expression(
    "c1*pow(pow(x[0]-V*t-ahead,2)+pow(x[1],2), 0.25)*(c2-cos(atan2(x[1],(x[0]-V*t-ahead))))*sin(0.5*atan2(x[1],(x[0]-V*t-ahead)))",
    degree=4,
    t=0,
    c1=c1,
    c2=c2,
    V=vel,
    ahead=ahead,
)


bcl = DirichletBC(V.sub(0), Constant(0.0), lefttop, method="pointwise")
bcb2 = DirichletBC(V.sub(1), r, bottom)
bct2 = DirichletBC(V.sub(1), r, top)
bcs = [bcl, bcb2, bct2]

bcb_du = DirichletBC(V.sub(1), Constant(0.0), bottom)
bct_du = DirichletBC(V.sub(1), Constant(0.0), top)
bcs_du = [bcl, bcb_du, bct_du]

cz = Constant(1.0)
bcb_z = DirichletBC(Y, cz, bottom)
bct_z = DirichletBC(Y, cz, top)
cz2 = Constant(0.0)
bcc_z = DirichletBC(Y, cz2, corner)
bcs_z = [bcb_z, bct_z, bcc_z]


########################################################################
# Define functions
########################################################################
du = TrialFunction(V)  # Incremental displacement
v = TestFunction(V)  # Test function
u = Function(V)  # Displacement from previous iteration
u_inc = Function(V)
dz = TrialFunction(Y)  # Incremental phase field
y = TestFunction(Y)  # Test function
z = Function(Y)  # Phase field from previous iteration
z_inc = Function(Y)
d = u.geometric_dimension()
B = Constant((0.0, 0.0))  # Body force per unit volume # NOTE: do we need it?
Tf = Expression(("t*0.0", "t*0"), degree=1, t=0)  # Traction force on the boundary NOTE: do we need it?

# ce_func = Function(Y)

##############################################################
# Initialisation of displacement field,u and the phase field,z
##############################################################
u_init = Constant((0.0, 0.0))
u.interpolate(u_init)
for bc in bcs:
    bc.apply(u.vector())

z_init = Constant(1.0)
z.interpolate(z_init)
for bc in bcs_z:
    bc.apply(z.vector())

z_ub = Function(Y)
z_ub.interpolate(Constant(1.0))
z_lb = Function(Y)
z_lb.interpolate(Constant(-0.0))


u_prev = Function(V)
assign(u_prev, u)
z_prev = Function(Y)
assign(z_prev, z)

z_stag = Function(Y)
assign(z_stag, z)

u_stag = Function(V)
assign(u_stag, u)


#################################################
###Label the dofs on boundary
#################################################
def extract_dofs_boundary(V, bsubd):
    label = Function(V)
    label_bc_bsubd = DirichletBC(V, Constant((1, 1)), bsubd)
    label_bc_bsubd.apply(label.vector())
    bsubd_dofs = np.where(label.vector() == 1)[0]
    return bsubd_dofs


# Dofs on which reaction is calculated
top_dofs = extract_dofs_boundary(V, top)
y_dofs_top = top_dofs[1::d]

# subdomains
tol = 1e-7
subdomain_0 = CompiledSubDomain("x[0] <= 2000 + tol", tol=tol)
subdomain_1 = CompiledSubDomain("x[0] >= 2000 + tol", tol=tol)
materials = MeshFunction("size_t", mesh, 2)
subdomain_0.mark(materials, 0)  # materials.set_all(0)
subdomain_1.mark(materials, 1)

boundary_subdomains = MeshFunction("size_t", mesh, 1)
boundary_subdomains.set_all(0)
left.mark(boundary_subdomains, 1)
right.mark(boundary_subdomains, 1)
bottom.mark(boundary_subdomains, 2)
top.mark(boundary_subdomains, 2)

# Define new measures associated with the interior domains
dx = dx(subdomain_data=materials)
ds = ds(subdomain_data=boundary_subdomains)


def energy(v):
    return (
        mu * (inner(sym(grad(v)), sym(grad(v))) + ((nu / (1 - nu)) ** 2) * (tr(sym(grad(v)))) ** 2)
        + 0.5 * (lmbda) * (tr(sym(grad(v))) * (1 - 2 * nu) / (1 - nu)) ** 2
    )
    # return (
    #     mu * (inner(sym(grad(v)), sym(grad(v)))) + 0.5 * (lmbda) * (tr(sym(grad(v)))) ** 2
    # )  # NOTE: the same with RACCOON W=0.5*K*tr(strain)^2 + G*dev(strain):dev(strain)


def epsilon(v):
    return sym(grad(v))


def sigma(v):
    # NOTE: sigma = lbda* I * tr(strain) + 2*mu*strain
    return 2.0 * mu * sym(grad(v)) + (lmbda) * tr(sym(grad(v))) * (1 - 2 * nu) / (1 - nu) * Identity(len(v))
    # return 2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(len(v))  # #TODO:


def sigmavm(sig, v):
    # return sqrt(3/2*(inner(sig-1/3*(1+nu)*tr(sig)*Identity(len(v)), sig-1/3*(1+nu)*tr(sig)*Identity(len(v))) + ((2*nu/3-1/3)**2)*tr(sig)**2 ))
    # return sqrt(
    #     3
    #     / 2
    #     * (
    #         inner(
    #             sig - 1 / 3 * tr(sig) * Identity(len(v)),
    #             sig - 1 / 3 * tr(sig) * Identity(len(v)),
    #         )
    #         + (1 / 9) * tr(sig) ** 2
    #     )
    # )
    return sqrt(
        1
        / 2
        * (
            inner(
                sig - 1 / 3 * tr(sig) * Identity(len(v)),
                sig - 1 / 3 * tr(sig) * Identity(len(v)),
            )
            + (1 / 9) * tr(sig) ** 2
        )
    )


# def sdev(sig, v):
#     return sig - 1 / 3 * tr(sig) * Identity(len(v))


# def sigma3(sdev):
#     return tr(sdev * sdev * sdev)


# Stored strain energy density (compressible L-P model)
psi1 = (z**2 + eta1) * (energy(u))  # NOTE: so psi1 is the degraded strian energy
psi11 = energy(u)  # NOTE: psi11 is the undegraded strain energy
stress = (z**2 + eta1) * sigma(u)

# ce = (beta1 * (z**2) * (tr(sigma(u))) + beta2 * (z**2) * (sigmavm(sigma(u), u)) + beta0) / (1 + beta3 * (z**4) * (tr(sigma(u))) ** 2)
## NOTE: 2024 ce formula
I1 = (z**2) * tr(sigma(u))
SQJ2 = (z**2) * sigmavm(sigma(u), u)
alpha1 = (delta * Gc) / (shs * 8 * eps) - (2 * Whs) / (3 * shs)
alpha2 = (3**0.5 * (3 * shs - sts) * delta * Gc) / (shs * sts * 8 * eps) + (2 * Whs) / (3**0.5 * shs) - (2 * 3**0.5 * Wts) / (sts)
ce = alpha2 * SQJ2 + alpha1 * I1 - z * (1 - sqrt(I1**2) / I1) * psi11

# Total potential energy
Pi = psi1 * dx(0)

# Compute first variation of Pi (directional derivative about u in the direction of v)
R1 = derivative(Pi, u, v)
R = R1

# Compute Jacobian of R
Jac1 = derivative(R1, u, du)
Jac = Jac1  # Jac, R -> u


# To use later for memory allocation for these tensors
A = PETScMatrix()
b = PETScVector()

# Balance of configurational forces PDE
Wv = pen / 2 * ((abs(z) - z) ** 2 + (abs(1 - z) - (1 - z)) ** 2) * dx(0)  # NOTE: penealty enforcing bounds (0, 1)
Wv2 = (
    # conditional(le(z, 0.05), 1, 0) * 10 * pen / 2 * (1 / 4 * (abs(z_prev - z) - (z_prev - z)) ** 2) * dx(0)
    conditional(le(z, 0.05), 1, 0)
    * 100
    * pen
    / 2
    * (1 / 4 * (abs(z_prev - z) - (z_prev - z)) ** 2)
    * dx(0)  # TODO: 2024
)  # NOTE: penalty enforcing conditional bounds

if phase_model == 1:
    # R_z = (
    #     y * 2 * z * (psi11) * dx(0)
    #     + y * (ce) * dx(0)
    #     + 3 * k / 8 * (y * (-1) / lch + 2 * lch * inner(grad(z), grad(y))) * dx(0)
    #     + derivative(Wv2, z, y)
    # )  # + derivative(Wv,z,y)  #linear model
    R_z = (
        y * 2 * z * (psi11) * dx
        + y * (ce) * dx
        + 3 * delta * Gc / 8 * (-y / eps + 2 * eps * inner(grad(z), grad(y))) * dx
        + derivative(Wv, z, y)
        # + derivative(Wv2, z, y)
    )
else:
    R_z = (
        y * 2 * z * (psi11 + ce) * dx(0) + Gc * (y * (z - 1) / lch + lch * inner(grad(z), grad(y))) * dx(0) + derivative(Wv2, z, y)
    )  # quadratic model

# Compute Jacobian of R_z
Jac_z = derivative(R_z, z, dz)


# Define the solver parameters

snes_solver_parameters = {
    "nonlinear_solver": "snes",
    "snes_solver": {
        # "linear_solver": "cg",  # lu or gmres or cg 'preconditioner: ilu, amg, jacobi'
        "linear_solver": "lu",
        "preconditioner": "ilu",
        # "maximum_iterations": 10,
        "maximum_iterations": 20,
        "report": True,
        "error_on_nonconvergence": False,
        # "error_on_nonconvergence": True
    },
}


# time-stepping parameters
T = 0.3
Totalsteps = 100
# minstepsize=1/Totalsteps/1000
# maxstepsize=1/Totalsteps*10
minstepsize = 1 / Totalsteps  # NOTE: I am forcing dt to be constant
maxstepsize = 1 / Totalsteps
startstepsize = 1 / Totalsteps
stepsize = startstepsize
t = stepsize
step = 1
samesizecount = 1
# other time stepping parameters
terminate = 0
printsteps = 1  # Number of incremental steps after which solution will be stored

u_inc1 = Function(V)
tau = 0
start_time = time.time()

# Solve variational problem
while t - stepsize < T:

    if comm_rank == 0:
        print("Step= %d" % step, "t= %f" % t, "Stepsize= %e" % stepsize)

    # c.t = t
    r.t = t

    stag_iter = 1
    unorm_stag = 1
    znorm_stag = 1
    rnorm_stag = 1
    while stag_iter < 800 and rnorm_stag > 1e-9:
    # while stag_iter < 800 and znorm_stag > 1e-7:
        ##############################################################
        # First PDE
        ##############################################################

        Problem_u = NonlinearVariationalProblem(R, u, bcs, J=Jac)
        solver_u = NonlinearVariationalSolver(Problem_u)
        solver_u.parameters.update(snes_solver_parameters)
        (iter, converged) = solver_u.solve()
        print(converged)

        ##############################################################
        # Second PDE
        ##############################################################
        start_time = time.time()
        Problem_z = NonlinearVariationalProblem(R_z, z, bcs_z, J=Jac_z)
        if phase_model == 1:
            Problem_z.set_bounds(z_lb, z_ub)
        solver_z = NonlinearVariationalSolver(Problem_z)
        solver_z.parameters.update(snes_solver_parameters)
        (iter, converged) = solver_z.solve()

        min_z = z.vector().min()
        zmin = MPI.min(comm, min_z)
        if comm_rank == 0:
            print(zmin)

        if comm_rank == 0:
            print("--- %s seconds ---" % (time.time() - start_time))

        ###############################################################
        # Residual check for stag loop
        ###############################################################
        b = assemble(-R, tensor=b)
        fint = b.copy()  # assign(fint,b)
        for bc in bcs_du:
            bc.apply(b)
        rnorm_stag = b.norm("l2")

        znorm = assemble(((z - z_stag) ** 2) * dx(0))
        znorm_stag = sqrt(znorm)

        unorm = assemble(((u - u_stag) ** 2) * dx(0))
        unorm_stag = sqrt(unorm)

        stag_iter += 1

        if comm_rank == 0:
            with open("./out/Stag_stopping_crit_" + str(step) + ".txt", "a") as rfile:
                rfile.write("%s %s\n" % (str(stag_iter), str(unorm_stag)))
                # rfile.write("%s %s\n" % (str(stag_iter), str(znorm_stag)))

        assign(z_stag, z)
        assign(u_stag, u)
    ######################################################################
    # Post-Processing
    if terminate == 1:
        assign(u, u_prev)
        assign(z, z_prev)
    else:
        assign(u_prev, u)
        assign(z_prev, z)

        tau += stepsize
        ####Calculate Reaction
        Fx = MPI.sum(comm, sum(fint[y_dofs_top]))
        # surfenergy = 3 / 8 * ((1 - z) / lch + lch * inner(grad(z), grad(z))) * dx(0)
        surfenergy = 3 / 8 * ((1 - z) / eps + eps * inner(grad(z), grad(z))) * dx(0)
        SE = assemble(surfenergy)

        JI1 = (psi1 - dot(dot(stress, n), u.dx(0))) * ds(1)
        JI2 = (-dot(dot(stress, n), u.dx(0))) * ds(2)
        Jintegral = assemble(JI1) + assemble(JI2)

        if comm_rank == 0:
            print(Fx)
            with open("./out/Surfing_nuc24_t_Jint.txt", "a") as rfile:
                rfile.write("%s %s\n" % (str(t), str(Jintegral)))

        ####Plot solution on incremental steps
        if step % printsteps == 0:
            file_results = XDMFFile("./out/surfing_nuc24_" + str(step) + ".xdmf")
            file_results.parameters["flush_output"] = True
            file_results.parameters["functions_share_mesh"] = True
            u.rename("u", "displacement field")
            z.rename("z", "phase field")
            file_results.write(u, step)
            file_results.write(z, step)

    # time stepping
    if terminate == 1:
        if stepsize > minstepsize:
            t -= stepsize
            stepsize /= 2
            t += stepsize
            samesizecount = 1
        else:
            break
    else:
        if samesizecount < 2:
            step += 1
            if t + stepsize <= T:
                samesizecount += 1
                t += stepsize
            else:
                samesizecount = 1
                stepsize = T - t
                t += stepsize

        else:
            step += 1
            if stepsize * 2 <= maxstepsize and t + stepsize * 2 <= T:
                stepsize *= 2
                t += stepsize
            elif stepsize * 2 > maxstepsize and t + maxstepsize <= T:
                stepsize = maxstepsize
                t += stepsize
            else:
                stepsize = T - t
                t += stepsize
            samesizecount = 1


#######################################################end of all loops
if comm_rank == 0:
    print("--- %s seconds ---" % (time.time() - start_time))

set_log_level(20)  # Default level
