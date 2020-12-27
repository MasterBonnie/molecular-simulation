from vpython import *
import numpy  as np
import helper
import static_state
scene = canvas(width=1200, height=800)
atom_radius = 0.3
scene.center = vector(0,0,0)
axes = [vector(1,0,0), vector(0,1,0), vector(0,0,1)]

dihedral = np.array([[2.4983673963624833, 3.9285080679660025, 0.49366912330355217 ],
[3.0953673963624833, 3.2804580679660025, 1.1509181233035521],
[2.253853396362483, 2.4933280679660026, 1.9924321233035522 ],
[1.7188973963624832, 3.1025280679660026, 2.5273881233035524]])

dihedrals = np.array([[0,1,2,3]])


i = dihedral[dihedrals[:,0]]
j = dihedral[dihedrals[:,1]]
k = dihedral[dihedrals[:,2]]
l = dihedral[dihedrals[:,3]]

C_1 = 0.94140 
C_2 = 2.82420 
C_3 = 0.0 
C_4 = -3.76560

f_i = helper.unit_vector(static_state.cross(i - j, k - j))
f_l = helper.unit_vector(static_state.cross(l - k, j - k))

theta_1 = helper.angle_between(i - j, k - j)
theta_2 = helper.angle_between(j - k, l - k)

o = (j+k)/2.0

psi = helper.angle_between(f_i, -f_l) - np.pi

magnitude = 0.5*(C_1*np.sin(psi) - 2*C_2*np.sin(2*psi) + 3*C_3*np.sin(3*psi) - 4*C_4*np.sin(4*psi))

force_i = (magnitude/(np.sin(np.pi - theta_1)*np.linalg.norm(i - j, axis=1)))[:, np.newaxis]*f_i
force_l = (magnitude/(np.sin(np.pi - theta_2)*np.linalg.norm(k - l, axis=1)))[:, np.newaxis]*f_l

force_k = static_state.cross( -1.0*(static_state.cross(i - j, force_i) + static_state.cross(l - j, force_l)), k - j)/((np.linalg.norm(k - j, axis=1)**2)[:, np.newaxis])
force_j = -(force_i + force_l + force_k)

force_i = helper.unit_vector(force_i)[0]
force_j = helper.unit_vector(force_j)[0]
force_k = helper.unit_vector(force_k)[0]
force_l = helper.unit_vector(force_l)[0]



scene.caption= """
To rotate "camera", drag with right button or Ctrl-drag.
To zoom, drag with middle button or Alt/Option depressed, or use scroll wheel.
  On a two-button mouse, middle is left + right.
To pan left/right and up/down, Shift-drag.
"""

atoms = [vector(pos[0], pos[1], pos[2]) for pos in dihedral]

#atoms = [vector(-0.9, 0.951, 0.309),
#                vector(-0.6, 0, 0),
#                vector(0.6, 0, 0),
#                vector(0.9, -0.454, 0.891)]

# atoms
for at in atoms:
  atom = sphere()
  atom.pos = at
  atom.radius = atom_radius
  atom.color = vector(0,0.58,0.69)

# Bonds
for i in range(3):
  curve(atoms[i], atoms[i+1],  radius=0.05)

lengthScaler = 0.8
pointer_i = arrow(pos=atoms[0], axis=vector(force_i[0], force_i[1], force_i[2]), shaftwidth=0.05, color=color.red,opacity=0.5)
pointer_j = arrow(pos=atoms[1], axis=vector(force_j[0], force_j[1], force_j[2]), shaftwidth=0.05, color=color.blue,opacity=0.5)
pointer_k = arrow(pos=atoms[2], axis=vector(force_k[0], force_k[1], force_k[2]), shaftwidth=0.05, color=color.green,opacity=0.5)
pointer_l = arrow(pos=atoms[3], axis=vector(force_l[0], force_l[1], force_l[2]), shaftwidth=0.05, color=color.purple,opacity=0.5)
