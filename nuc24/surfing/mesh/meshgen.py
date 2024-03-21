import gmsh

def surf_mesh(xlim: list[float], ylim: list[float], h: float, mesh_path: str):
    gmsh.initialize()
    gmsh.model.add(f"Surfing")

    # nx, ny for transfinite
    nx = int(abs(xlim[1] - xlim[0]) / h) + 1
    ny = int(abs(ylim[1] - ylim[0]) / h) + 1

    # points
    p1 = gmsh.model.geo.addPoint(xlim[0], ylim[0], 0, h)
    p2 = gmsh.model.geo.addPoint(xlim[1], ylim[0], 0, h)
    p3 = gmsh.model.geo.addPoint(xlim[1], ylim[1], 0, h)
    p4 = gmsh.model.geo.addPoint(xlim[0], ylim[1], 0, h)

    # lines
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # curve loop and plane surface
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    ps = gmsh.model.geo.addPlaneSurface([cl])

    # transfinite
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, nx)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, ny)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx)
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, ny)
    gmsh.model.geo.mesh.setTransfiniteSurface(ps, cornerTags=[p1, p2, p3, p4])
    # gmsh.model.geo.mesh.setRecombine(2, ps)

    # physical groups for boundaries
    # gmsh.model.addPhysicalGroup(1, [l4], name="left")
    # gmsh.model.addPhysicalGroup(1, [l2], name="right")
    # gmsh.model.addPhysicalGroup(1, [l3], name="top")
    # gmsh.model.addPhysicalGroup(1, [l1], name="bottom")
    gmsh.model.addPhysicalGroup(2, [cl], name="surfing")

    # synchronize
    gmsh.model.geo.synchronize()

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)

    # save to msh and fenics xdmf
    gmsh.write(f"{mesh_path}.msh")

    # Optional: Launch the Gmsh GUI to view the mesh
    # gmsh.fltk.run()
    gmsh.finalize()
  
# mesh gen
mesh_path = "./surf_tri"
xlim = [0, 30]
ylim = [-5, 5]
h = 0.1
surf_mesh(xlim, ylim, h, mesh_path)