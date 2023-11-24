import gamspy
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum, Sense
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    n = 8
    m = 5
    D_1 = np.random.binomial(10, 0.5, n)
    print("D1: ", D_1)
    D_2 = np.random.binomial(10, 0.5, n)
    print("D2: ", D_2)
    
    coeff = np.random.randint(10, size=m)
    print("coeff: ", coeff)
    s     = np.array([1,1,1,1,1])
    print("s: ", s)
    l_q   = np.random.randint(10, size=n)
    print("l-q= ", l_q)
    matrix_A= np.zeros(m*n).reshape(m, n)
    for i in range(m):
        for j in range(n):
            matrix_A[i][j] = np.random.randint(5)
    '''
    print("input coefficience of objective function")
    for i in range (n):
        coeff[i] = f.readline()

    print("input right-hand side")
    for i in range (m):
        rhs[i] = f.readline()
    
    print("input matrix")
    for i in range (m):
        for j in range (n):
            matrix_A[i][j] = f.readline()
    '''
    n_index = np.arange(n)
    m_index = np.arange(m)
    K = np.array([0, 1])
    
    model = Container()
    i = Set(container=model, name='index_of_variables', records=n_index)
    j = Set(container=model, name='index_of_constraints', records=m_index)
    #k = Set(container=model, name='scenarios', records=K)
    S = Parameter(
        container=model,
        name='S',
        domain=j,
        description='savage',
        records=s,
    )
    L_Q = Parameter(
        container=model,
        name='LQ',
        domain=i,
        description='LQ',
        records=l_q,
    )
    d1 = Parameter(
        container=model,
        name='d1',
        domain=i,
        description='demand_1',
        records=D_1,
    )
    d2 = Parameter(
        container=model,
        name='d2',
        domain=i,
        description='demand_2',
        records=D_2,
    )
    c = Parameter(
        container=model,
        name='c',
        domain=j,
        description='coefficient of objective function',
        records=coeff,
    )
    A = Parameter(
        container=model,
        name="A",
        domain=[j, i],
        description="matrix A",
        records=matrix_A,
    )   
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
    z1_constraints = Equation(
        container=model, name="z1_constraints", domain=i, description="z1constraints"
    )
    z1_constraints[i] = z1[i] <= d1[i]
    z2_constraints = Equation(
        container=model, name="z2_constraints", domain=i, description="z2constraints"
    )
    z2_constraints[i] = z2[i] <= d2[i]
    y1_constraints = Equation(
        container=model, name="y1_constraints", domain=j, description="c1"
    )
    y1_constraints[j] = y1[j] == x[j] - Sum(i, A[j, i]*z1[i])
    y2_constraints = Equation(
        container=model, name="y2_constraints", domain=j, description="c2"
    )
    y2_constraints[j] = y2[j] == x[j] - Sum(i, A[j, i]*z2[i])
    obj = Sum(j, c[j]*x[j]) + 0.5*(Sum(i, L_Q[i] * z1[i]) - Sum(j, S[j] * y1[j])) + 0.5*(Sum(i, L_Q[i] * z2[i]) - Sum(j, S[j] * y2[j]))
    transport = Model(
        model,
        name="linear_programming",
        equations=[z1_constraints, z2_constraints, y1_constraints, y2_constraints],
        problem="LP",
        sense=Sense.MIN,
        objective=obj,
    )   
    transport.solve(output=sys.stdout)
    print(transport.objective_value)

if __name__ == '__main__':
    main()