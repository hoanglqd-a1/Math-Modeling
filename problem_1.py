from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Sense
import sys
import numpy as np

def main():
    n = 8
    m = 5
    #2 scenarios D
    D1 = np.random.binomial(10, 0.5, n)
    D2 = np.random.binomial(10, 0.5, n)
    print(D1)
    print(D2)
    #vector b: preorder cost bj per unit of part j
    b = np.array([2, 2, 3, 4, 5])
    #vector s: savage value where sj < bj
    s = np.array([1, 1, 1, 1, 1])
    #vector c: c = (ci:= li − qi) are cost coefficients
    c = np.array([-44, -47, -44, -45, -48, -47, -42, -46])
    #matrix A: a unit of product i requires aij units of part j
    matrix_A=np.array([ [2., 4., 3., 4., 4.],
                        [0., 0., 4., 0., 3.],
                        [1., 1., 1., 0., 2.],
                        [0., 2., 1., 0., 4.],
                        [1., 3., 0., 3., 3.],
                        [4., 2., 4., 0., 4.],
                        [0., 1., 0., 0., 3.],
                        [0., 4., 3., 3., 0.]])
    #build model
    set_n = np.arange(n)
    set_m = np.arange(m)
    
    model = Container()
    i = Set(container=model, name='set_n', records=set_n)
    j = Set(container=model, name='set_m', records=set_m)
    #Parameters
    B = Parameter(
        container=model, name='b', domain=j, description='preorder cost', records=b,
    )
    S = Parameter(
        container=model, name='S', domain=j, description='savage cost', records=s,
    )
    C = Parameter(
        container=model, name='C', domain=i, description='cost coefficents', records=c,
    )
    d1 = Parameter(
        container=model, name='d1', domain=i, description='demand_1', records=D1,
    )
    d2 = Parameter(
        container=model, name='d2', domain=i, description='demand_2', records=D2,
    )
    A = Parameter(
        container=model,
        name="A",
        domain=[i, j],
        description="matrix A",
        records=matrix_A,
    )
    #Decision variable x, y1, y2, z1, z2
    x = Variable(
        container=model,
        name="x",
        domain=j,
        type="Positive",
        description="x_decision variable",
)
    y1 = Variable(
        container=model,
        name='y1',
        domain=j,
        type='Positive',
        description='y1_decision variable',
    )
    y2 = Variable(
        container=model,
        name='y2',
        domain=j,
        type='Positive',
        description='y2_decision variable',
    )
    z1 = Variable(
        container=model,
        name='z1',
        domain=i,
        type='Positive',
        description='z1_decision variable',
    )
    z2 = Variable(
        container=model,
        name='z2',
        domain=i,
        type='Positive',
        description='z2_decision variable',
    )
    #constraints
    z1_ctr = Equation(
        container=model, name="z1_constraints", domain=i, description="z1ctr"
    )
    z1_ctr[i] = z1[i] <= d1[i]
    z2_ctr = Equation(
            container=model, name="z2_constraints", domain=i, description="z2ctr"
    )
    z2_ctr[i] = z2[i] <= d2[i]
    y1_ctr = Equation(
            container=model, name="y1_constraints", domain=j, description="c1ctr"
    )
    y1_ctr[j] = y1[j] == x[j] - Sum(i, A[i, j]*z1[i])
    y2_ctr = Equation(
            container=model, name="y2_constraints", domain=j, description="c2ctr"
    )
    y2_ctr[j] = y2[j] == x[j] - Sum(i, A[i, j]*z2[i])
    #Objective function
    obj = Sum(j, B[j]*x[j]) + 0.5*(Sum(i, C[i] * z1[i]) - Sum(j, S[j] * y1[j])) + 0.5*(Sum(i, C[i] * z2[i]) - Sum(j, S[j] * y2[j]))
    transport = Model(
        model,
        name="linear_programming",
        equations=[z1_ctr, z2_ctr, y1_ctr, y2_ctr],
        problem="LP",
        sense=Sense.MIN,
        objective=obj,
    )   
    #Solve the optimization problem
    transport.solve(output=sys.stdout)
    print(transport.objective_value)

if __name__ == '__main__':
    main()
