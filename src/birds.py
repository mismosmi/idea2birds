################################################################################
#                       idea2birds                                             #
#                   Bird simulation in python                                  #
################################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
import time

width, height, n = 640, 480, 500

def limit(target, upperbound=False, lowerbound=False):
# Multiplies Vector by a factor so that lowerbound < |v| < upperbound
    norm = np.sqrt((target*target).sum(axis=1)).reshape(n,1)
    if upperbound:
        np.multiply(target, upperbound/norm, out=target, where=norm > upperbound)
    if lowerbound:
        np.multiply(target, lowerbound/norm, out=target, where=norm < lowerbound)



class Flock:
    def __init__(self, angle_view=30, max_velocity=1., max_acceleration=0.03):
        # typecast for command line argument input + angle deg->rad
        angle_view, max_velocity, max_acceleration = float(angle_view)/180*np.pi, float(max_velocity), float(max_acceleration)
        # minimum velocity, basically only matters for birds that lose their flocks.
        min_velocity = 0.5

        # randomly distributed starting velocities, might come out pretty much the same as if you just set self.velocity = np.zeros((n,2),dtype=np.float32) but avoids division by zero.
        angles = np.random.uniform(0,2*np.pi,n)
        self.velocity = (np.array([np.cos(angles),np.sin(angles)])*np.random.uniform(min_velocity,max_velocity,n)).astype(np.float32).T
        self.position = np.random.rand(n,2).astype(np.float32)*[width,height]
        #self.position = np.array([[width/2+15,height/2],[width/2-15,height/2]],dtype=np.float32)


        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.max_acceleration = max_acceleration
        self.angle_view = angle_view


    def run(self):
        self.distance = self.calc_distance() 

        
        # check if viewing-angle (theta_velocity) += angle_view/2 matches distance-vector to neighbors (theta(pos1-pos2))
        mask_view = np.absolute(np.arctan2(self.dy, self.dx) - np.arctan2(self.velocity[:,1],self.velocity[:,0])) < self.angle_view/2
        mask_0 = (self.distance > 0)
        mask_1 = (self.distance < 25)
        mask_2 = (self.distance < 50)
        mask_1 *= mask_0*mask_view
        mask_2 *= mask_0*mask_view
        mask_3 = mask_2

        mask_1_count = np.maximum(mask_1.sum(axis=1), 1).reshape(n,1)
        mask_2_count = np.maximum(mask_2.sum(axis=1), 1).reshape(n,1)
        mask_3_count = mask_2_count

        separation = self.calc_separation(mask_1,mask_1_count)
        alignment = self.calc_alignment(mask_2,mask_2_count)
        cohesion = self.calc_cohesion(mask_3,mask_3_count)


        acceleration = 1.5 * separation + alignment + cohesion
        #acceleration = 1.5 * separation
        #acceleration = alignment + cohesion
        #acceleration = alignment
        #acceleration = cohesion
        self.velocity += acceleration
        self.random_turn()
        limit(self.velocity, self.max_velocity, self.min_velocity)
        self.position += self.velocity
        self.position %= [width, height]


    def calc_distance(self):
        """
        Calculate distance between birds. Return n x n array with each row
        corresponding to each bird's distance to the others
        """
        # subtract each element of the position array from the others
        self.dx = np.subtract.outer(self.position[:, 0], self.position[:, 0])
        self.dy = np.subtract.outer(self.position[:, 1], self.position[:, 1])

        # wrap around
        np.subtract(self.dx, width+1, out=self.dx, where=self.dx > width/2)
        np.subtract(self.dy, height+1, out=self.dy, where=self.dy > height/2)
        np.add(width+1, self.dx, out=self.dx, where=-self.dx > width/2)
        np.add(height+1, self.dy, out=self.dy, where=-self.dy > height/2)

        # return hypotenuse of corresponding elements i.e. distance
        return np.hypot(self.dx, self.dy)

    def calc_alignment(self, mask, count):
        # Compute the average velocity of local neighbours
        target = np.dot(mask, self.velocity)/count

        # Compute steering
        norm = np.sqrt((target*target).sum(axis=1)).reshape(n, 1)
        target = self.max_velocity * np.divide(target, norm, out=target, where=norm != 0)
        target -= self.velocity

        limit(target, self.max_acceleration)
        return target


    def calc_cohesion(self, mask, count):
        # Compute the gravity center of local neighbours
        target = np.dot(mask, self.position)/count
        
        # Compute direction toward the center
        target -= self.position
        
        # Normalize the result
        norm = np.sqrt((target*target).sum(axis=1)).reshape(n, 1)
        target = self.max_velocity * np.divide(target, norm, out=target, where=norm != 0)
        
        # Compute the resulting steering
        target -= self.velocity

        limit(target, self.max_acceleration)
        return target


    def calc_separation(self, mask, count):
        # Compute the repulsion force from local neighbours
        repulsion = np.dstack((self.dx, self.dy))
        
        # Force is inversely proportional to the distance
        np.divide(repulsion, self.distance.reshape(n, n, 1)**2, out=repulsion, where=self.distance.reshape(n, n, 1) != 0)
        
        # Compute direction away from others
        target = (repulsion*mask.reshape(n, n, 1)).sum(axis=1)/count
        
        # Normalize the result
        norm = np.sqrt((target*target).sum(axis=1)).reshape(n, 1)
        target = self.max_velocity * np.divide(target, norm, out=target, where=norm != 0)
        
        # Compute the resulting steering
        target -= self.velocity

        limit(target, self.max_acceleration)
        return target

    def random_turn(self):
        angles = np.random.normal(0,0.1,n)
        c = np.cos(angles)
        s = np.sin(angles)
        self.velocity[:,0] = self.velocity[:,0]*c - self.velocity[:,1]*s
        self.velocity[:,1] = self.velocity[:,1]*c + self.velocity[:,0]*s
        return
        



class MarkerCollection:
    """
    Marker collection
    """

    def __init__(self, n=100):
        v = np.array([(-0.25, -0.25), (+0.0, +0.5), (+0.25, -0.25), (0, 0)])
        c = np.array([Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
        self._base_vertices = np.tile(v.reshape(-1), n).reshape(n, len(v), 2)
        self._vertices = np.tile(v.reshape(-1), n).reshape(n, len(v), 2)
        self._codes = np.tile(c.reshape(-1), n)

        self._scale = np.ones(n)
        self._translate = np.zeros((n, 2))
        self._rotate = np.zeros(n)

        self._path = Path(vertices=self._vertices.reshape(n*len(v), 2),
                          codes=self._codes)
        self._collection = PathCollection([self._path], linewidth=0.5,
                                          facecolor="k", edgecolor="w")

    def update(self):
        n = len(self._base_vertices)
        self._vertices[...] = self._base_vertices * self._scale
        cos_rotate, sin_rotate = np.cos(self._rotate), np.sin(self._rotate)
        R = np.empty((n, 2, 2))
        R[:, 0, 0] = cos_rotate
        R[:, 1, 0] = sin_rotate
        R[:, 0, 1] = -sin_rotate
        R[:, 1, 1] = cos_rotate
        self._vertices[...] = np.einsum('ijk,ilk->ijl', self._vertices, R)
        self._vertices += self._translate.reshape(n, 1, 2)



def update(*args):
    global flock, collection, trace

    # Flock updating
    flock.run()
    collection._scale = 10
    collection._translate = flock.position
    collection._rotate = -np.pi/2 + np.arctan2(flock.velocity[:, 1],
                                               flock.velocity[:, 0])
    collection.update()

    # Trace updating
    if trace is not None:
        P = flock.position.astype(int)
        trace[height-1-P[:, 1], P[:, 0]] = .75
        trace *= .99
        im.set_array(trace)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    
    argc = len(sys.argv)
    if argc > 4:
        n = int(sys.argv[4])
        if argc > 5:
            width = int(sys.argv[5])
            if argc > 6:
                height = int(sys.argv[6])
    if argc == 1:
        print('Command Line Arguments: angle_view max_velocity max_acceleration n width height')

    flock = Flock(*sys.argv[1:4])
    fig = plt.figure(figsize=(10, 10*height/width), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], aspect=1, frameon=False)
    collection = MarkerCollection(n)
    ax.add_collection(collection._collection)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks([])
    ax.set_yticks([])

    # Trace
    trace = None
    if 0:
        trace = np.zeros((height, width))
        im = ax.imshow(trace, extent=[0, width, 0, height], vmin=0, vmax=1,
                       interpolation="nearest", cmap=plt.cm.gray_r)

    animation = FuncAnimation(fig, update, interval=10, frames=1000)
    #animation.save('boid.mp4', fps=40, dpi=80, bitrate=-1, codec="libx264",
    #               extra_args=['-pix_fmt', 'yuv420p'],
    #               metadata={'artist': 'Nicolas P. Rougier'})
    plt.show()

