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
import argparse

width, height, n = 640,480,500

def limit(target, upperbound=False, lowerbound=False):
# Multiplies Vector by a factor so that lowerbound < |v| < upperbound
    norm = np.sqrt((target*target).sum(axis=1)).reshape(n,1)
    if upperbound:
        np.multiply(target, upperbound/norm, out=target, where=norm > upperbound)
    if lowerbound:
        np.multiply(target, lowerbound/norm, out=target, where=np.logical_and(norm < lowerbound, norm != 0))



class Flock:
    def __init__(self, args):

        # randomly distributed starting velocities, might come out pretty much the same as if you just set self.velocity = np.zeros((n,2),dtype=np.float32) but avoids division by zero.
        angles = np.random.uniform(0,2*np.pi,n)
        self.velocity = (np.array([np.cos(angles),np.sin(angles)])*args.v).astype(np.float32).T
        self.position = np.random.rand(n,2).astype(np.float32)*[width,height]

        # test boid configuration
        if n<=3:
            self.position = np.array([[width/2+15,height/2],[width/2-15,height/2], [width/2,height/2+15]],dtype=np.float32)
            self.velocity = np.array([[self.args.v,0],[-self.args.v,0],[self.args.v,0]],dtype=np.float32)
            self.position = self.position[0:n,:]
            self.velocity = self.velocity[0:n,:]

        self.args = args

        if args.angle == 0:
            self.angle_view = False
        else:
            # half of viewing angle + deg->rad
            self.angle_view = args.angle/360*np.pi

        
    def get_va(self):
        if self.args.v:
            vges = np.sum(self.velocity,axis=0)
            return 1/(n*self.args.v) * np.sqrt((vges*vges).sum())

    def run(self):
        self.distance = self.calc_distance() 
        
        mask = (self.distance < self.args.radius)

        # check if viewing-angle (theta_velocity) +- angle_view/2 matches distance-vector to neighbors (theta(pos1-pos2))
        if self.angle_view:
            mask *= np.absolute(np.arctan2(self.dy.T, self.dx.T) - np.arctan2(self.velocity[:,1],self.velocity[:,0])) < self.angle_view

        mask_count = np.maximum(mask.sum(axis=1), 1).reshape(n,1)
    
        self.calc_velocity(mask,mask_count)

        if  self.args.eta:
            self.random_turn()

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
        np.subtract(self.dx, np.sign(self.dx)*width, out=self.dx, where=np.absolute(self.dx) > width/2)
        np.subtract(self.dy, np.sign(self.dy)*height, out=self.dy, where=np.absolute(self.dy) > height/2)

        # return hypotenuse of corresponding elements i.e. distance
        return np.hypot(self.dx, self.dy)


    def calc_velocity(self, mask, count):
        # Compute the average velocity of local neighbours
        target = np.dot(mask, self.velocity)/count

        # Compute steering
        norm = np.sqrt((target*target).sum(axis=1)).reshape(n, 1)
        target = self.args.v * np.divide(target, norm, out=target, where=norm != 0)

        self.velocity = target
        return


    def random_turn(self):
        angles = np.random.uniform(-self.args.eta/2,self.args.eta/2,n)
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
    global flock, collection, trace, param_record

    # record parameters
    if param_record:
        param_record["va"][args[0]] = flock.get_va()
            

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

    
defaults = argparse.Namespace()
defaults.angle = 0
defaults.v = 3.
defaults.width = 500
defaults.height = 500
defaults.n = 100
defaults.radius = 100
defaults.frames = 1000
defaults.vfile = "birds.mp4"
defaults.fps = 40
defaults.eta = 0

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", "-a", help="Boid field of Vision [deg], default="+str(defaults.angle), type=float, default=defaults.angle)
    parser.add_argument("-v", help="Velocity, default="+str(defaults.v), type=float, default=defaults.v)
    parser.add_argument("--width", help="Range for x-coordinate, default="+str(defaults.width), type=int, default=defaults.width)
    parser.add_argument("--height", help="Range for y-coordinate, default="+str(defaults.height),type=int, default=defaults.height)
    parser.add_argument("--n", "-n", help="Number of boids, default="+str(defaults.n),type=int, default=defaults.n)
    parser.add_argument("--eta", help="Generates Angles -eta/2 < delta Theta < eta/2, default="+str(defaults.eta), type=float, default=defaults.eta)
    parser.add_argument("--radius", "-r", help="Viewing Radius, default="+str(defaults.radius), type=int, default=defaults.radius)
    parser.add_argument("--export", "-e", help="Export video file and exit", action="store_true")
    parser.add_argument("--frames", help="Number of frames for export to video or parameter record file, default="+str(defaults.frames), type=int, default=defaults.frames)
    parser.add_argument("--vfile", help="Out-File for video export, default="+str(defaults.vfile), type=str, default=defaults.vfile)
    parser.add_argument("--fps", help="Set Video Framerate for export, default="+str(defaults.fps), type=int, default=defaults.fps)
    parser.add_argument("--scale", help="Scale field size.", type=float)
    parser.add_argument("--out", "-o", help="Specify output File for recording Parameters, Filetype: .npz", type=str)
    parser.add_argument("--batch", "-b", help="Batch mode: no live display, no video export", action="store_true")




    args = parser.parse_args()


    if args.scale:
        width = int(args.scale * args.width)
        height = int(args.scale * args.height)
        n = args.n
    else:
        width, height, n = args.width, args.height, args.n

    
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

        

    if args.out:
        # initialize storage arrays
        param_record = {
            "frames": args.frames,
            "va": np.zeros(args.frames),
            "eta": args.eta,
            "rho": n/(width*height),
            "v": args.v if args.v else [args.min_velocity,args.max_velocity]
            }

        # run loop
        if args.batch:
            for i in range(0,args.frames):
                va[i] = flock.get_va()
                flock.run()
    else:
        param_record = False
        



    if not args.batch:
        if args.export:
            animation = FuncAnimation(fig, update, interval=10, frames=args.frames)
            animation.save(args.vfile, fps=args.fps, dpi=80, bitrate=-1, codec="libx264",
                       extra_args=['-pix_fmt', 'yuv420p'],
                       metadata={'artist': 'Nicolas P. Rougier'})

        else:
            animation = FuncAnimation(fig, update, interval=10, frames=args.frames, repeat=not args.out)
            plt.show()

    if args.out:
        # save to file
        np.savez(args.out, **param_record)



