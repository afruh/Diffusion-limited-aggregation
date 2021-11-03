import ParticleClass as pc
import numpy as np

import os
from matplotlib.collections import EllipseCollection
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
np.random.seed(99)

NumberOfMonomers, NumberOfAggregate = 200,1
L_xMin, L_xMax = 0, 30
L_yMin, L_yMax = 0, 30
NumberMono_per_kind = np.array([1,199])
Radiai_per_kind = np.array([0.5,0.5])
Densities_per_kind = np.array([1,1])
bond_length_scale = 1.5
k_BT = 10
# call constructor, which should initialize the configuration
mols = pc.Aggregate(NumberOfMonomers, NumberOfAggregate, L_xMin, L_xMax, L_yMin,
                     L_yMax, NumberMono_per_kind, Radiai_per_kind,
                     Densities_per_kind, bond_length_scale, k_BT)

# we could initialize next_event, but it's not necessary
# next_event = pc.CollisionEvent( Type = 'wall or other, to be determined',
# dt = 0, mono_1 = 0, mono_2 = 0, w_dir = 0)

'''define parameters for MD simulation'''
t = 0.0
dt = 0.02
NumberOfFrames = 3000
next_event = mols.compute_next_event()

Agg=[]
T=[]

def MolecularDynamicsLoop(frame):
    '''
    The MD loop including update of frame for animation.
    '''

    global t, mols, next_event
    if len(mols.aggregate)==mols.NM:
        ani.event_source.stop()
    ti=time.time()
    Agg.append(len(mols.aggregate))


    next_frame_t = t + dt

    while t + next_event.dt < next_frame_t:
        mols.pos += mols.vel * next_event.dt
        t += next_event.dt
        mols.compute_new_velocities(next_event)
        next_event = mols.compute_next_event()
    dt_remaining = next_frame_t - t
    mols.pos += mols.vel * dt_remaining
    t += dt_remaining
    next_event = mols.compute_next_event()

    # ec=0
    # for k in range(mols.NM):
    #     ec+=mols.mass[k]*np.linalg.norm(mols.vel[k])**2/2
    # Ec.append(ec)

    plt.title(f'$t = {t}$, remaining frames = {NumberOfFrames-(frame+1)}')
    MonomerColors[mols.aggregate] = 0
    collection.set_array(MonomerColors)
    collection.set_offsets(mols.pos)
    collection_aggregate.set_offsets(mols.pos[mols.aggregate])
    T.append(ti-time.time())
    return collection

'''We define and initalize the plot for the animation'''
fig, ax = plt.subplots()
L_xMin, L_yMin = mols.BoxLimMin  # not defined if initalized by file
L_xMax, L_yMax = mols.BoxLimMax  # not defined if initalized by file
BorderGap = 0.1*(L_xMax - L_xMin)
ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)
ax.set_aspect('equal')

# confining hard walls plotted as dashed lines
rect = mpatches.Rectangle((L_xMin, L_yMin), L_xMax-L_xMin, L_yMax-L_yMin,
                          linestyle='dashed', ec='gray', fc='None')
ax.add_patch(rect)

MonomerColors = np.ones(mols.NM)*0.1  # unique color for monomers
MonomerColors[mols.aggregate] = 0

Width, Hight, Angle = mols.bond_length, mols.bond_length, np.zeros( mols.NA )
collection_aggregate = EllipseCollection( Width, Hight, Angle, units='x', offsets=mols.pos[mols.aggregate],
               transOffset=ax.transData, edgecolor = 'k', facecolor = 'None')
ax.add_collection(collection_aggregate)
# plotting all monomers as solid circles
Width, Hight, Angle = 2*mols.rad, 2*mols.rad, np.zeros(mols.NM)
collection = EllipseCollection(Width, Hight, Angle, units='x', offsets=mols.pos,
                               transOffset=ax.transData, cmap='tab10',
                               edgecolor='k')
collection.set_array(MonomerColors)
collection.set_clim(0, 1)  # <--- we set the limit for the color code
ax.add_collection(collection)



"""
Create the animation, i.e. looping NumberOfFrames
over the update function
"""
Delay_in_ms = 40  # dely between images/frames for plt.show()
ani = FuncAnimation(fig, MolecularDynamicsLoop, frames=NumberOfFrames,
                    interval=Delay_in_ms, blit=False, repeat=False)
plt.show()


frames=np.arange(len(T))
plt.figure()
plt.plot(frames,T)
plt.figure()
plt.plot(frames,Agg,'k')
plt.show()

