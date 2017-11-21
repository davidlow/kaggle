# Make a support vector machine framework

# Theory: From wikipedia
# For a trainng set (x,y), ... (x_n,y_n), x vectors,
#
# Find the maximul-margin hyperplane that divides the points x_i for y_i=1
# from the points x_j for y_j=-1. 
#
# Any hyperplane can be written as the set of points x satisfying 
#   dot(w,x)-b = 0
# where w is the normal vector to the hyperplane.  b/abs(w) is the 
# offset of the hyperplane from the origin along the normal vector w.
#
#

