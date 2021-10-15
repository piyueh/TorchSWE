Note that, theoretically speaking, the boundary condition for water depth at
x=25 is that h=0.66 when the flow is subcritical, and when flow is
supercritical, we use the outflow boundary condition.

However, the solver has not had time-dependent boundary conditions like this. So
we use h=0.66 all the way in this example. The result will have mostly correct
values in the domain but a wrong solution at the last cell.
