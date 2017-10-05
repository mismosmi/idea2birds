################################################################################
#                       idea2birds                                             #
#                   Bird simulation in python                                  #
################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
import argparse

class Flock:
    def __init__(self, args):
        n = args.n
        self.velocity = np.zeros((n,2),dtype=np.float32)
        self.position = np.zeros((n,2),dtype=np.float32)
        self.r = 10  # Radius of influence
        self.max_velocity = 1.0
        self.mu = args.mu 

        self.position[:, 0] = np.random.uniform(0, width, n)
        self.position[:, 1] = np.random.uniform(0, height, n)
        self.angles = np.random.uniform(0, 2*np.pi, n)
        self.velocity[:, 0] = np.cos(self.angles) * self.max_velocity
        self.velocity[:, 1] = np.sin(self.angles) * self.max_velocity
        
        # test boid configuration
        if n<=3:
            self.position = np.array([[width/2+15,height/2],[width/2-15,height/2], [width/2,height/2+15]],dtype=np.float32)
            self.velocity = np.array([[0.5,0],[-0.5,0],[0.5,0]],dtype=np.float32)
            self.position = self.position[0:n,:]
            self.velocity = self.velocity[0:n,:]


    def run(self):
        self.velocity = self.calc_velocities()
        self.position += self.velocity

        # Wrap around
        self.position += [width, height]
        self.position %= [width, height]

    def calc_distance(self):
        """
        Calculate distance between birds. Return n x n array with each row
        corresponding to each bird's distance to the others
        """
        # subtract each element of the position array from the others
        dx = np.subtract.outer(self.position[:, 0], self.position[:, 0])
        dy = np.subtract.outer(self.position[:, 1], self.position[:, 1])

        # return hypotenuse of corresponding elements i.e. distance
        return np.hypot(dx, dy)

    def calc_velocities(self):
        """
        Calculate updated velocities. New angles are calculated as
        theta = arctan( <\sin(theta_r)> / <cos(theta_r)>
        with theta_r being radius in which bird is influenced by others
        """
        distance = self.calc_distance()

        # Masks tell whether or not a condition is met for a given element
        # Result is an array containing True or False
        mask_zero = (0 <= distance)  # Only look at positive distances
        mask_radius = (distance < self.r)  # Only fetch birds within radius r
        mask = mask_zero * mask_radius
        count = mask.sum(axis=1)

        # Get average values of direction
        cos_avg = np.sum(mask * np.cos(self.angles), axis=1) / count
        sin_avg = np.sum(mask * np.sin(self.angles), axis=1) / count

        # Compute new angles
        angles_avg = np.arctan2(sin_avg, cos_avg)
        #angles_avg += np.pi*(cos_avg < 0)
        if self.mu:
            angles_avg += np.random.uniform(-self.mu, self.mu, len(angles_avg))

        self.angles = angles_avg

        # Update velocities
        velocities = np.zeros((n, 2), dtype=float)
        velocities[:, 0] = np.cos(angles_avg)
        velocities[:, 1] = np.sin(angles_avg)

        return velocities * self.max_velocity
        


        
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

    defaults = {
        "n": 300,
        "mu": .1
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu",help="Maximum angle for uniform distribution of random turns,default="+str(defaults["mu"]),type=float, default=defaults["mu"])
    parser.add_argument("--n","-n",help="Number of birds, default="+str(defaults["n"]),type=int, default=defaults["n"])

    args = parser.parse_args()

    n = args.n    
    width, height = 640, 360
    flock = Flock(args)
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

    animation = FuncAnimation(fig, update, interval=10, frames=10)
    # animation.save('boid.mp4', fps=40, dpi=80, bitrate=-1, codec="libx264",
    #                extra_args=['-pix_fmt', 'yuv420p'],
    #                metadata={'artist': 'Nicolas P. Rougier'})
    plt.show()

