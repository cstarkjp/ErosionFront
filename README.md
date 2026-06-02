# ErosionFront

<div align="center">
<h3> Geomorphic Hamiltonian theory of rock slope erosion and the emergent geometry of Richter slopes
 </h3>
</div>

<div align="center">

![Blender view of West Mitten Butte](Images/WestMittenButte/BlenderView1_reduced.png)

</div>

 An iconic image of the American West is the mesa: a steep cliff, rising above a ramp-like rock slope, capped by a flat bench. This famous landform is thought to develop where strong rock overlies weak, and where rockfall debris buffers ramp erosion. However, this explanation is unnecessarily complicated. We instead argue that the archetypal geometry is an emergent property that arises even in uniform bedrock with no talus armouring. This assertion springs from a phenomenological model of scarp retreat, in which the combined speed of weathering and surface-normal erosion is a slowly varying function of gradient. Model analysis, using geometric mechanics and level sets, reveals the ramp-cliff transition to form automatically as a shock solution of a non-convex Hamilton-Jacobi equation (HJE). Erodibility contrasts are not needed to explain this behaviour, but when present they help lock the landform into its classic shape and allow it to persist long-term. Our theoretical conclusions are vindicated by 3D topographic analysis of differential cliff recession in geologically homogeneous material.


 ![Animated set of HJE solutions of ramp-cliff retreat for varying ratio of upper/lower rock layer erodibility](Simulation/Combo/combined_reversed.png?raw=true)

 ### Level-set solution of a geomorphic HJE

The purpose of the Python code presented here is to derive, analyze, and numerically solve a geomorphic Hamiltonian[^1] model of rock slope erosion and retreat[^2]. The code is provided as a 
[Python library package](Packages/erosionfront)
 and associated Jupyter notebooks (e.g., [here](Simulation/ErosionFront.ipynb) and  [here](Analysis/3DProfiling.ipynb)).

Numerical solution of the model Hamilton-Jacobi equation is achieved with a level-set scheme[^3] that employs Lax-Friedrichs finite differencing to obtain stable viscosity solutions for a non-convex Hamiltonian. The level-set code is custom implemented in Python and exploits [`scikit-fmm`](https://pythonhosted.org/scikit-fmm/) fast-marching for speedy computation of distance functions.

Model analysis is performed using geometric mechanics[^4]: having converted the rock-slope erosion model into geomorphic Hamiltonian $\mathcal{H}(\mathbf{p},\mathbf{r})$ form, this Hamiltonian is then used to derive Hamilton's ray tracing equations $(\partial_{\mathbf{p}}\mathcal{H}, -\partial_{\mathbf{r}}\mathcal{H})$ and the co-metric tensor $g^{ij} = \partial_{ij}\mathcal{H}$; these properties are then probed to understand model stability, notably to place bounds on the non-convexity of $\mathcal{H}$ and to identify critical angles.

[^1]: [Stark, C.P., & Stark, G.J., 2022. The direction of landscape erosion. Earth Surface Dynamics, 10: 383-419.](https://doi.org/10.5194/esurf-10-383-2022)

[^2]: [Howard, A.D., & Selby, M.J., 2009. Rock Slopes. In: Parsons, A.J., Abrahams, A.D. (eds). Geomorphology of Desert Environments. Springer, Dordrecht. ](https://doi.org/10.1007/978-1-4020-5719-9_8)

[^3]: [Osher, S., & Fedkiw, R., 2003. Level Set Methods and Dynamic Implicit Surfaces. Springer-Verlag New York, Inc.](https://link.springer.com/book/10.1007/b98879)  See page 50.

[^4]: [Holm, D.D., 2011. Geometric Mechanics. Part I: Dynamics and Symmetry (2nd Edition)](https://www.ma.imperial.ac.uk/~dholm/classnotes/HolmPart1-GM.pdf)