import numpy as np
from matplotlib.collections import EllipseCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pickle


class CollisionEvent:
    """
    Object contains all information about a collision event
    which are necessary to update the velocity after the collision.
    For MD of hard spheres (with hard bond-length dimer interactions)
    in a rectangular simulation box with hard walls, there are only
    two distinct collision types:
    1) wall collision of particle i with vertical or horizontal wall
    2) external (or dimer bond-length) collision between particle i and j
    """

    def __init__(self, Type='wall or other', dt=np.inf, mono_1=0, mono_2=0,
                 w_dir=1):
        """
        Type = 'wall' or other
        dt = remaining time until collision
        mono_1 = index of monomer
        mono_2 = if inter-particle collision, index of second monomer
        w_dir = if wall collision, direction of wall
        (   w_dir = 0 if wall in x direction, i.e. vertical walls
            w_dir = 1 if wall in y direction, i.e. horizontal walls   )
        """
        self.Type = Type
        self.dt = dt
        self.mono_1 = mono_1
        self.mono_2 = mono_2  # only importent for interparticle collisions
        self.w_dir = w_dir  # only important for wall collisions

    def __str__(self):
        if self.Type == 'wall':
            return (f"Event type: {self.Type}, dt: {self.dt},"
                    f" p1 = {self.mono_1}, dim = {self.w_dir}")
        else:
            return (f"Event type: {self.Type}, dt: {self.dt},"
                    f" p1 = {self.mono_1}, p2 = {self.mono_2}")


class Monomers:
    """
    Class for event-driven molecular dynamics simulation of hard spheres:
    -Object contains all information about a two-dimensional monomer system
    of hard spheres confined in a rectengular box of hard walls.
    -A configuration is fully described by the simulation box and
    the particles positions, velocities, radiai, and masses.
    -Initial configuration of $N$ monomers has random positions (without
    overlap) and velocities of random orientation and norms satisfying
    $E = sum_i^N m_i / 2 (v_i)^2 = N d/2 k_B T$, with $d$ being the dimension,
    $k_B$ the Boltzmann constant, and $T$ the temperature.
    -Class contains all functions for an event-driven molecular dynamics (MD)
    simulation. Essentail for inter-particle collsions is the mono_pair array,
    which book-keeps all combinations without repetition of particle-index
    pairs for inter-particles collisions, e.g. for $N = 3$ particles
    indices = 0, 1, 2
    mono_pair = [[0,1], [0,2], [1,2]]
    -Monomers can be initialized with individual radiai and
    density = mass/volume.
    For example:
    NumberOfMonomers = 7
    NumberMono_per_kind = [ 2, 5]
    Radiai_per_kind = [ 0.2, 0.5]
    Densities_per_kind = [ 2.2, 5.5]
    then monomers mono_0, mono_1 have radius 0.2 and mass 2.2*pi*0.2^2
    and monomers mono_2,...,mono_6 have radius 0.5 and mass 5.5*pi*0.5^2
    -The dictionary of this class can be saved in a pickle file at any time of
    the MD simulation. If the filename of this dictionary is passed to
    __init__ the simulation can be continued at any point in the future.
    IMPORTANT! If system is initialized from file, then other parameters
    of __init__ are ignored!
    """

    def __init__(self, NumberOfMonomers=4, L_xMin=0, L_xMax=1, L_yMin=0,
                 L_yMax=1, NumberMono_per_kind=np.array([4]),
                 Radiai_per_kind=0.5*np.ones(1), Densities_per_kind=np.ones(1),
                 k_BT=1, FilePath='./Configuration.p'):
        try:
            self.__dict__ = pickle.load(open(FilePath, "rb"))
            print(f"IMPORTANT! System is initialized from file {FilePath}, "
                  "i.e. other input parameters of __init__ are ignored!")
        except IOError:
            assert(NumberOfMonomers > 0)
            assert((L_xMin < L_xMax) and (L_yMin < L_yMax))
            self.NM = NumberOfMonomers
            self.DIM = 2  # dimension of system
            self.BoxLimMin = np.array([L_xMin, L_yMin])
            self.BoxLimMax = np.array([L_xMax, L_yMax])
            # Masses, negative mass means not initialized
            self.mass = -1*np.ones(self.NM)
            # Radiai, negative radiai means not initialized
            self.rad = -1*np.ones(self.NM)
            # Positions, not initalized but desired shape
            self.pos = np.empty((self.NM, self.DIM))
            # Velocities, not initalized but desired shape
            self.vel = np.empty((self.NM, self.DIM))
            self.mono_pairs = np.array([(k, l) for k in range(self.NM)
                                        for l in range(k+1, self.NM)])
            self.next_wall_coll = CollisionEvent('wall', np.inf, 0, 0, 0)
            self.next_mono_coll = CollisionEvent('mono', np.inf, 0, 0, 0)

            self.assignRadiaiMassesVelocities(NumberMono_per_kind,Radiai_per_kind,Densities_per_kind,k_BT)
            self.assignRandomMonoPos()

    def save_configuration(self, FilePath='MonomerConfiguration.p'):
        '''Saves configuration. Callable at any time during simulation.'''
        # print( self.__dict__ )

    def assignRadiaiMassesVelocities(self, NumberMono_per_kind=np.array([4]),
                                     Radiai_per_kind=0.5*np.ones(1),
                                     Densities_per_kind=np.ones(1), k_BT=1):
        '''
        Make this a PRIVATE function -> cannot be called outside class
        definition.
        '''
        '''initialize radiai and masses'''
        assert(sum(NumberMono_per_kind) == self.NM)
        assert(isinstance(Radiai_per_kind, np.ndarray)
               and (Radiai_per_kind.ndim == 1))
        assert((Radiai_per_kind.shape == NumberMono_per_kind.shape)
               and (Radiai_per_kind.shape == Densities_per_kind.shape))


        l_nbr = len(NumberMono_per_kind)
        current_index = 0
        for k in range(l_nbr):
            for i in range(NumberMono_per_kind[k]):
                self.rad[current_index] = Radiai_per_kind[k]
                self.mass[current_index] = (Densities_per_kind[k] * np.pi
                                            * Radiai_per_kind[k]**2)
                current_index += 1

        '''initialize velocities'''
        assert(k_BT > 0)


        #first particle which is the aggregate with smal velocity
        theta=np.random.uniform(0,2*np.pi)
        V0=1e-1
        self.vel[0,:]=[V0*np.cos(theta),V0*np.sin(theta)]

        E_kin=self.NM*self.DIM/2*k_BT - self.mass[0]*V0**2/2
        Vstat=[]
        for i in range(1,self.NM-1):
            vmax=np.sqrt(2*E_kin/self.mass[i])
            v=np.random.uniform(0,vmax/4)
            Vstat.append(v)
            theta=np.random.uniform(0,2*np.pi)
            E_kin-=self.mass[i]/2*v**2
            self.vel[i,:]=[v*np.cos(theta),v*np.sin(theta)]
        #last particle : random not used for the norm of v
        v=np.sqrt(2*E_kin/self.mass[-1])
        Vstat.append(v)
        theta=np.random.uniform(0,2*np.pi)
        self.vel[-1,:]=[v*np.cos(theta),v*np.sin(theta)]


        # for i in range(self.NM):
        #     print(np.linalg.norm(self.vel[i,:]))
        # print('moy,e-t: ',np.mean(Vstat),np.std(Vstat))




    def assignRandomMonoPos(self, start_index=0):
        '''
        Make this a PRIVATE function -> cannot be called outside class
        definition.
        Initialize random positions without overlap between monomers and wall.
        '''
        assert (min(self.rad) > 0)  # otherwise not initialized
        mono_new, infiniteLoopTest = start_index, 0
        BoxLength = self.BoxLimMax - self.BoxLimMin
        while mono_new < self.NM and infiniteLoopTest < 10**4:
            infiniteLoopTest += 1
            x = np.random.uniform(self.BoxLimMin[0] + self.rad[mono_new],
                                  self.BoxLimMax[0] - self.rad[mono_new])
            y = np.random.uniform(self.BoxLimMin[1] + self.rad[mono_new],
                                  self.BoxLimMax[1] - self.rad[mono_new])
            new_pos = np.array([x, y])
            overlap = False
            for i in range(mono_new):
                if (np.linalg.norm(new_pos - self.pos[i]) < self.rad[mono_new]
                        + self.rad[i]):
                    overlap = True
                    break
            if not overlap:
                self.pos[mono_new] = new_pos
                mono_new += 1
        if mono_new != self.NM:
            print('Failed to initialize all particle positions.'
                  '\nIncrease simulation box size!')
            exit()

    def __str__(self, index='all'):
        if index == 'all':
            return (f"\nMonomers with:\nposition = {self.pos}\nvelocity ="
                    f" {self.vel}\nradius = {self.rad}\nmass = {self.mass}")
        else:
            return (f"\nMonomer at index = {index} with:\nposition = "
                    f"{self.pos[index]}\nvelocity = {self.vel[index]}\nradius"
                    f" = {self.rad[index]}\nmass = {self.mass[index]}")

    def Wall_time(self):
        '''
        -Function computes list of remaining time dt until future
        wall collision in x and y direction for every particle.
        Then, it stores collision parameters of the event with
        the smallest dt in the object next_wall_coll.
        -Meaning of future:
        if v > 0: solve BoxLimMax - rad = x + v * dt
        else:     solve BoxLimMin + rad = x + v * dt
        '''

        collision_list = np.zeros((self.NM, self.DIM))
        collision_dt = np.zeros((self.NM, self.DIM))
        # Calculate all collision times
        for i in range(self.DIM):
            collision_list_max = (self.BoxLimMax[i] - self.rad)
            collision_list_min = (self.BoxLimMin[i] + self.rad)
            collision_list[:, i] = np.where(self.vel[:, i] > 0,
                                            collision_list_max,
                                            collision_list_min)
            collision_dt[:, i] = ((collision_list[:, i] - self.pos[:, i])
                                  / self.vel[:, i])

        c_mins_index = np.where(collision_dt == np.min(collision_dt))
        minCollTime = collision_dt[c_mins_index][0]
        collision_disk = c_mins_index[0][0]
        wall_direction = c_mins_index[1][0]

        self.next_wall_coll.dt = minCollTime
        self.next_wall_coll.mono_1 = collision_disk
        # self.next_wall_coll.mono_2 = not necessary
        self.next_wall_coll.w_dir = wall_direction

    def Mono_pair_time(self):
        '''
        - Function computes list of remaining time dt until
        future external collition between all combinations of
        monomer pairs without repetition. Then, it stores
        collision parameters of the event with
        the smallest dt in the object next_mono_coll.
        - If particles move away from each other, i.e.
        scal >= 0 or Omega < 0, then remaining dt is infinity.
        '''

        mono_i = self.mono_pairs[:, 0]  # List of collision partner 1
        mono_j = self.mono_pairs[:, 1]  # List of collision partner 2

        delta_pos= np.zeros((len(mono_i),self.DIM))
        delta_vel= np.zeros((len(mono_i),self.DIM))

        List_dt=np.zeros(len(mono_i))
        A = np.zeros(len(mono_i))
        B = np.zeros(len(mono_i))
        Omega = np.zeros(len(mono_i))

        delta_pos = self.pos[mono_i,:]-self.pos[mono_j,:]
        delta_vel = self.vel[mono_i,:]-self.vel[mono_j,:]

        A = delta_vel[:,0]**2+delta_vel[:,1]**2
        B = 2*delta_vel[:,0]*delta_pos[:,0] + 2*delta_vel[:,1]*delta_pos[:,1]
        Omega = B**2 - 4*A*( delta_pos[:,0]**2+delta_pos[:,1]**2 - (self.rad[mono_i] + self.rad[mono_j])**2)
        sol_m = 1/(2*A)*(-B - np.sqrt(Omega))
        List_dt=np.where(np.logical_or(Omega<0,B>=0),np.inf,sol_m)

        Index=np.argmin(List_dt)


        self.next_mono_coll.dt = List_dt[Index]
        self.next_mono_coll.mono_1 = mono_i[Index]
        self.next_mono_coll.mono_2 = mono_j[Index]
    def compute_next_event(self):
        '''
        Function gets event information about:
        1) next possible wall event
        2) next possible pair event
        Function returns event info of event with
        minimal time, i.e. the clostest in future.
        '''

        self.Wall_time()
        self.Mono_pair_time()
        if self.next_wall_coll.dt < self.next_mono_coll.dt:
            return self.next_wall_coll
        else:
            return self.next_mono_coll


    def snapshot(self, FileName='./snapshot.png', Title='$t = $?'):
        '''
        Function saves a snapshot of current configuration,
        i.e. particle positions as circles of corresponding radius,
        velocities as arrows on particles,
        blue dashed lines for the hard walls of the simulation box.
        '''
        fig, ax = plt.subplots(dpi=300)
        L_xMin, L_xMax = self.BoxLimMin[0], self.BoxLimMax[0]
        L_yMin, L_yMax = self.BoxLimMin[1], self.BoxLimMax[1]
        BorderGap = 0.1*(L_xMax - L_xMin)
        ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
        ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)

        # --->plot hard walls (rectangle)
        rect = mpatches.Rectangle((L_xMin, L_yMin), L_xMax-L_xMin,
                                  L_yMax-L_yMin, linestyle='dashed',
                                  ec='gray', fc='None')
        ax.add_patch(rect)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ position')
        ax.set_ylabel('$y$ position')

        # --->plot monomer positions as circles
        MonomerColors = np.linspace(0.2, 0.95, self.NM)
        Width, Hight, Angle = 2*self.rad, 2*self.rad, np.zeros(self.NM)
        collection = EllipseCollection(Width, Hight, Angle, units='x',
                                       offsets=self.pos,
                                       transOffset=ax.transData,
                                       cmap='nipy_spectral', edgecolor='k')
        collection.set_array(MonomerColors)
        collection.set_clim(0, 1)  # <--- we set the limit for the color code
        ax.add_collection(collection)

        # --->plot velocities as arrows
        ax.quiver(self.pos[:, 0], self.pos[:, 1], self.vel[:, 0],
                  self.vel[:, 1], units='dots', scale_units='dots')

        plt.title(Title)
        plt.savefig(FileName)
        plt.close()

class Aggregate(Monomers):
    """
    --> Class derived from Monomers and Dimers.
    --> See also comments in Monomer class and Dimer class.
    --> Class for event-driven molecular dynamics simulation of hard-sphere
    system with the aggregate (and monomers).
    -Monomers can be initialized with individual radiai and density = mass/volume.
    For example:
    NumberOfMonomers = 10
    NumberOfDimers = 2
    bond_length_scale = 1.2
    NumberMono_per_kind = [ 2, 2, 6]
    Radiai_per_kind = [ 0.2, 0.5, 0.1]
    Densities_per_kind = [ 2.2, 5.5, 1.1]
    then monomers mono_0, mono_1 have radius 0.2 and mass 2.2*pi*0.2^2
    and monomers mono_2, mono_3 have radius 0.5 and mass 5.5*pi*0.5^2
    and monomers mono_4,..., mono_9 have radius 0.1 and mass 1.1*pi*0.1^2
    dimer pairs are: (mono_0, mono_2), (mono_1, mono_3) with bond length 1.2*(0.2+0.5)
    see bond_length_scale and radiai
    -The dictionary of this class can be saved in a pickle file at any time of
    the MD simulation. If the filename of this dictionary is passed to
    __init__ the simulation can be continued at any point in the future.
    IMPORTANT! If system is initialized from file, then other parameters
    of __init__ are ignored!
    """
    def __init__(self, NumberOfMonomers = 4, NumberOfAggregate = 1, L_xMin = 0, L_xMax = 1, L_yMin = 0, L_yMax = 1, NumberMono_per_kind = np.array([1,3]), Radiai_per_kind = 0.5*np.ones(2), Densities_per_kind = np.array([10,1]), bond_length_scale = 1.2, k_BT = 1, FilePath = './Configuration.p'):
        #if __init__() defined in derived class -> child does NOT inherit parent's __init__()
        try:
            self.__dict__ = pickle.load( open( FilePath, "rb" ) )
            print("IMPORTANT! System is initialized from file %s, i.e. other input parameters of __init__ are ignored!" % FilePath)
        except:
            assert ( (NumberOfAggregate > 0) and (NumberOfMonomers >= 2*NumberOfAggregate) )
            assert ( bond_length_scale > 1. ) # is in units of minimal distance of respective monomer pair
            Monomers.__init__(self, NumberOfMonomers, L_xMin, L_xMax, L_yMin, L_yMax, NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )
            self.NA = NumberOfAggregate
            self.ND=0
            self.dimer_pairs = np.array([])
            self.aggregate = np.arange(self.NA)
            self.bond_length = np.array([2*bond_length_scale*self.rad[0]])
            self.next_dimer_coll = CollisionEvent( 'dimer', 0, 0, 0, 0)

            '''
            Positions initialized as pure monomer system by monomer __init__.
            ---> Reinitalize all monomer positions, but place aggregate first.
            '''
            self.pos[0]=[(L_xMax-L_xMin)/2,(L_yMax-L_yMin)/2]
            self.assignRandomMonoPos(1)


    def __str__(self, index = 'all'):
        if index == 'all':
            return Monomers.__str__(self) + "\ndimer pairs = " + str(self.dimer_pairs) + "\nwith max bond length = " + str(self.bond_length)
        else:
            return "\nDimer pair " + str(index) + " consists of monomers = " + str(self.dimer_pairs[index]) + "\nwith max bond length = " + str(self.bond_length[index]) + Monomers.__str__(self, self.dimer_pairs[index][0]) + Monomers.__str__(self, self.dimer_pairs[index][1])

    def Dimer_pair_time(self):
        '''
        Function computes list of remaining time dt until
        future dimer bond collition for all dimer pairs.
        Then, it stores collision parameters of the event with
        the smallest dt in the object next_dimer_coll.
        '''
        if len(self.dimer_pairs)==0:
            self.next_dimer_coll.dt=np.inf
            return
        mono_i = self.dimer_pairs[:, 0]  # List of collision partner 1
        mono_j = self.dimer_pairs[:, 1]  # List of collision partner 2

        delta_x0 = self.pos[mono_i, 0] - self.pos[mono_j, 0]
        delta_y0 = self.pos[mono_i, 1] - self.pos[mono_j, 1]

        delta_vx = self.vel[mono_i, 0] - self.vel[mono_j, 0]
        delta_vy = self.vel[mono_i, 1] - self.vel[mono_j, 1]

        a = delta_vx * delta_vx + delta_vy * delta_vy
        b = 2*(delta_vx*delta_x0 + delta_vy*delta_y0)
        c = (delta_x0**2 + delta_y0**2
             - (self.bond_length[0])**2)

        Omega = b**2 - 4*a*c
        Omega = np.where(np.logical_and((Omega > 0),(b > 0)), Omega, np.inf)

        dt_collision_minus = 1/(2*a)*(-b - np.sqrt(Omega))
        dt_collision_plus = 1/(2*a)*(-b + np.sqrt(Omega))
        dt_collision = np.array([dt_collision_minus, dt_collision_plus])
        dt_collision_index = np.where(dt_collision > 0,
                                      dt_collision,
                                      np.inf)
        dt_collision_index = np.where(dt_collision_index
                                      == np.min(dt_collision_index))

        minCollTime = dt_collision[dt_collision_index[0][0],
                                   dt_collision_index[1][0]]
        if minCollTime < 1e-13:
            minCollTime = [np.inf]
        collision_disk_1 = mono_i[dt_collision_index[1]][0]
        collision_disk_2 = mono_j[dt_collision_index[1]][0]

        self.next_dimer_coll.dt = minCollTime

        self.next_dimer_coll.mono_1 = collision_disk_1
        self.next_dimer_coll.mono_2 = collision_disk_2

        '''
        There are actually no conditions on the collision time.
        It is in principle guaranteed that c <= 0 and b**2-4*a*c >= 0
        But that requires that c <= 0.
        '''
        CollTime=np.where(np.logical_and((b**2-4*a*c)>0 ,c<=0),(-b+np.sqrt(b**2-4*a*c))/(2*a),np.inf)
        pair_min=np.argmin(CollTime)
        self.next_dimer_coll.dt = CollTime[pair_min]
        self.next_dimer_coll.Type='dimer'
        self.next_dimer_coll.mono_1 = self.dimer_pairs[pair_min,0]
        self.next_dimer_coll.mono_2 = self.dimer_pairs[pair_min,1]

    def compute_next_event(self):
        '''
            Function gets event information about:
            1) next possible wall event
            2) next possible pair event
            Function returns event info of event with
            minimal time, i.e. the clostest in future.
            '''
        self.Wall_time()
        self.Mono_pair_time()
        self.Dimer_pair_time()

        if self.next_mono_coll.dt>self.next_wall_coll.dt and self.next_dimer_coll.dt>self.next_wall_coll.dt:
            return self.next_wall_coll
        elif self.next_mono_coll.dt>self.next_dimer_coll.dt:
            return self.next_dimer_coll
        else:
            return self.next_mono_coll


    def compute_new_velocities(self, next_event):
        '''
        Function updates the velocities of the monomer(s)
        involved in collision event.
        Update depends on event type.
        Ellastic wall collisions in x direction reverse vx.
        Ellastic pair collisions follow:
        https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
        '''

        if next_event.Type == "wall":
            self.vel[next_event.mono_1, next_event.w_dir] *= -1
        else:
            delta = (np.array(self.pos[next_event.mono_2]
                     - self.pos[next_event.mono_1]))
            delta_hat = delta / np.linalg.norm(delta)
            delta_v = (np.array(self.vel[next_event.mono_1]
                       - self.vel[next_event.mono_2]))
            self.vel[next_event.mono_1] = (self.vel[next_event.mono_1]
                                           - (2*self.mass[next_event.mono_2]
                                           / (self.mass[next_event.mono_1]
                                              + self.mass[next_event.mono_2]))
                                           * delta_hat @ np.transpose(delta_v)
                                           * delta_hat)
            self.vel[next_event.mono_2] = (self.vel[next_event.mono_2]
                                           + (2*self.mass[next_event.mono_1]
                                           / (self.mass[next_event.mono_1]
                                              + self.mass[next_event.mono_2]))
                                           * delta_hat @ np.transpose(delta_v)
                                           * delta_hat)

            #si mono_1 est un des dimers il faut transformer masse et vitesse de 2:
            if next_event.mono_1 in self.aggregate and not(next_event.mono_2 in self.aggregate):
                self.aggregation(next_event.mono_1,next_event.mono_2)

            #si mono_2 est un des dimers il faut transformer masse et vitesse de 1:
            if next_event.mono_2 in self.aggregate and not(next_event.mono_1 in self.aggregate):
                self.aggregation(next_event.mono_2,next_event.mono_1)


    def aggregation(self,aggregate, mono):
        '''Function applies the transformation below when there is a collision aggregate monomers:
         1)create a new dimer ( and the bond correponding)
         2)update the velocity norm of the new aggregated particle
         3)update the mass of the new aggregated particle ( m_orange --> m_blue)
         '''

        if len(self.dimer_pairs)==0:
            self.dimer_pairs=np.array([[aggregate,mono]])
        else:
            self.dimer_pairs = np.append(self.dimer_pairs,[[aggregate,mono]],axis=0)
            self.bond_length=np.append( self.bond_length,self.bond_length[0])
            self.vel[mono] *= np.sqrt(self.mass[mono]/self.mass[aggregate])
            self.mass[mono] = self.mass[aggregate]
            self.aggregate=np.append(self.aggregate,[mono])


    def snapshot(self, FileName = './snapshot.png', Title = ''):
        '''
        ---> Overwriting snapshot(...) of Monomers class!
        Function saves a snapshot of current configuration,
        i.e. monomer positions as circles of corresponding radius,
        dimer bond length as back empty circles (on top of monomers)
        velocities as arrows on monomers,
        blue dashed lines for the hard walls of the simulation box.
        '''
        fig, ax = plt.subplots( dpi=300 )
        L_xMin, L_xMax = self.BoxLimMin[0], self.BoxLimMax[0]
        L_yMin, L_yMax = self.BoxLimMin[1], self.BoxLimMax[1]
        BorderGap = 0.1*(L_xMax - L_xMin)
        ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
        ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)

        #--->plot hard walls (rectangle)
        rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin, linestyle='dashed', ec='gray', fc='None')
        ax.add_patch(rect)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ position')
        ax.set_ylabel('$y$ position')

        #--->plot monomer positions as circles
        COLORS = np.linspace(0.2,0.95,self.ND+1)
        MonomerColors = np.ones(self.NM)*COLORS[-1] #unique color for monomers
        # recolor each monomer pair with individual color
        MonomerColors[self.dimer_pairs[:,0]] = COLORS[:len(self.dimer_pairs)]
        MonomerColors[self.dimer_pairs[:,1]] = COLORS[:len(self.dimer_pairs)]

        #plot solid monomers
        Width, Hight, Angle = 2*self.rad, 2*self.rad, np.zeros( self.NM )
        collection = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
        collection.set_array(MonomerColors)
        collection.set_clim(0, 1) # <--- we set the limit for the color code
        ax.add_collection(collection)

        #plot bond length of dimers as black cicles
        Width, Hight, Angle = self.bond_length, self.bond_length, np.zeros( self.ND )
        mono_i = self.dimer_pairs[:,0]
        mono_j = self.dimer_pairs[:,1]
        collection_mono_i = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos[mono_i],
                       transOffset=ax.transData, edgecolor = 'k', facecolor = 'None')
        collection_mono_j = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos[mono_j],
                       transOffset=ax.transData, edgecolor = 'k', facecolor = 'None')
        ax.add_collection(collection_mono_i)
        ax.add_collection(collection_mono_j)

        #--->plot velocities as arrows
        ax.quiver( self.pos[:,0], self.pos[:,1], self.vel[:,0], self.vel[:,1] , units = 'dots', scale_units = 'dots')

        plt.title(Title)
        plt.savefig( FileName)
        plt.close()
