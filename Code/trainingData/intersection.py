def sphere_line_intersection(l1, u, sp, r):

    def square(f):
        return f * f
    from math import sqrt
    from numpy import linalg as LA
    import numpy as np

    # l1[0],l1[1],l1[2]  P1 coordinates (point of line)
    # l2[0],l2[1],l2[2]  P2 coordinates (point of line)
    # sp[0],sp[1],sp[2], r  P3 coordinates and radius (sphere)
    # x,y,z   intersection coordinates
    #
    # This function returns a pointer array which first index indicates
    # the number of intersection point, followed by coordinate pairs.
  
    #p1 = p2 = 0 
    p1 = p2 =  D1 = D2 = None
    D = 0
    a = square(u[0]) + square(u[1]) + square(u[2])
    b = 2.0 * ((u[0]) * (l1[0] - sp[0]) +
               (u[1]) * (l1[1] - sp[1]) +
               (u[2]) * (l1[2] - sp[2]))

    c = (square(sp[0]) + square(sp[1]) + square(sp[2]) + square(l1[0]) +
            square(l1[1]) + square(l1[2]) -
            2.0 * (sp[0] * l1[0] + sp[1] * l1[1] + sp[2] * l1[2]) - square(r))

    i = b * b - 4.0 * a * c

    if i < 0.0:
        p1 = p2 = D = 0
        #pass  # no intersections
    elif i == 0.0:
        # one intersection
        p[0] = 1.0

        mu = -b / (2.0 * a)
        p1 = (l1[0] + mu * (u[0]),
              l1[1] + mu * (u[1]),
              l1[2] + mu * (u[2]),
              )
        v1 = np.array(p1)-l1 
        if (np.sum((v1 >= 0 )*1-(u>= 0)*1)) == 0:
          D = LA.norm(np.array(v1))
        else:
          D = 0
    elif i > 0.0:
        # first intersection
        #print('two intersections')
        mu = (-b + sqrt(i)) / (2.0 * a)
        p1 = (l1[0] + mu * (u[0]),
              l1[1] + mu * (u[1]),
              l1[2] + mu * (u[2]),
              )
        v1 = np.array(p1)-l1 
        if (np.sum((v1 >= 0 )*1-(u>= 0)*1)) == 0:
          D1 = LA.norm(np.array(v1))
          D = D1 
          #print('True 1')
        # second intersection
        mu = (-b - sqrt(i)) / (2.0 * a)
        p2 = (l1[0] + mu * (u[0]),
              l1[1] + mu * (u[1]),
              l1[2] + mu * (u[2]),
              )
        v1 = np.array(p2)-l1
        if (np.sum((v1 >= 0 )*1-(u>= 0)*1)) == 0:

          D2 = LA.norm(np.array(v1))
          if D1 != None:
            D = np.min([D,D2])
          else:
            D = D2
          # print('true 2')

    return D

import numpy as np
# u = np.array([1,1,0])
# l1 = np.array([0,0,0])
# print(type(l1))
# sp = [2,2,0]
# r = 1
# p1, p2 ,D = sphere_line_intersection(l1, u, sp, r)
# print(p1,p2,D)