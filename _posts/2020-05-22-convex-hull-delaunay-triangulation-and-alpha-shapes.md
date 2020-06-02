---
title: Convex hull, Delaunay triangulation and Alpha Shapes
---

In this article, I am going to present what is a convex hull, what is its purpose and how can we construct it in Python. Then I enter into more details by defining an Alpha Shape and using Delaunay triangulation method to construct it.

Convex Hull.
The main idea behind a convex hull comes from the definition of a convex polygon. As a reminder, a convex polygon is a polygon with all its interior angles less than 180^degrees. By using this definition, we figure out that every regular polygon is convex. If one angle has more than 180 degrees, the polygon is considered to be concave.
A convex hull uses the same principle as convex polygon applied to set of points. For instance, a convex hull is the smallest convex polygon containing all the points of a set.
One of the purpose of a convex hull is to prune a specific area in a plane. It is mostly used in computer graphics, geometry and navigation. One interesting use case of convex hull is presented in some papers presenting techniques to construct floorplans based on data collected from phone sensors. By using convex hull, we can prune a plan and focus only on the area containing the collected traces.


Alpha Shapes.
