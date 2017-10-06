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
        np.multiply(target, lowerbound/norm, out=target, where=norm < lowerbound)



class Flock:
    def __init__(self, args):

        # randomly distributed starting velocities, might come out pretty much the same as if you just set self.velocity = np.zeros((n,2),dtype=np.float32) but avoids division by zero.
        angles = np.random.uniform(0,2*np.pi,n)
        self.velocity = (np.array([np.cos(angles),np.sin(angles)])*np.random.uniform(args.min_velocity,args.max_velocity,n)).astype(np.float32).T
        self.position = np.random.rand(n,2).astype(np.float32)*[width,height]

        # test boid configuration
        if n<=3:
            self.position = np.array([[width/2+15,height/2],[width/2-15,height/2], [width/2,height/2+15]],dtype=np.float32)
            self.velocity = np.array([[0.5,0],[-0.5,0],[0.5,0]],dtype=np.float32)
            self.position = self.position[0:n,:]
            self.velocity = self.velocity[0:n,:]

        if not args.v and args.max_velocity == args.min_velocity:
            args.v = args.max_velocity

        self.args = args
        if args.angle == 0:
            self.angle_view = False
        else:
            # half of viewing angle + deg->rad
            self.angle_view = args.angle/360*np.pi
        if args.alignment_radius == args.cohesion_radius:
            self.r_cohesion = False
        else:
            self.r_cohesion = args.cohesion_radius

        self.angle_alignment, self.angle_cohesion, self.angle_separation = "alignment" in args.limit_view, "cohesion" in args.limit_view, "separation" in args.limit_view

        
    def get_va(self):
        if self.args.v:
            vges = np.sum(self.velocity,axis=0)
            return 1/(n*self.args.v) * np.sqrt((vges*vges).sum())
        else:
            vges = np.sum(self.velocity,axis=0)
            nges = np.sum(np.sqrt((self.velocity*self.velocity).sum(axis=1)))
            return (vges*vges).sum()/nges

    def run(self):
        self.distance = self.calc_distance() 

        mask_0 = (self.distance > 0)
        mask_1 = (self.distance < self.args.separation_radius)
        mask_1 *= mask_0
        mask_1_count = np.maximum(mask_1.sum(axis=1), 1).reshape(n,1)

        # check if viewing-angle (theta_velocity) +- angle_view/2 matches distance-vector to neighbors (theta(pos1-pos2))
        if self.angle_view:
            mask_view = np.absolute(np.arctan2(self.dy.T, self.dx.T) - np.arctan2(self.velocity[:,1],self.velocity[:,0])) < self.angle_view
            mask_0 *= mask_view
            
        mask_2 = (self.distance < self.args.alignment_radius)
        mask_2 *= mask_0

        mask_2_count = np.maximum(mask_2.sum(axis=1), 1).reshape(n,1)
        if self.r_cohesion:
            mask_3 = (self.distance < self.r_cohesion)
            mask_3_count = np.maximum(mask_3.sum(axis=1), 1).reshape(n,1)
        else:
            mask_3 = mask_2
            mask_3_count = mask_2_count

        separation = self.calc_separation(mask_1,mask_1_count)
        alignment = self.calc_alignment(mask_2,mask_2_count)
        cohesion = self.calc_cohesion(mask_3,mask_3_count)


        acceleration = self.args.separation * separation + self.args.alignment * alignment + self.args.cohesion * cohesion
        self.velocity += acceleration
        if self.args.random:
            self.random_turn()
        limit(self.velocity, self.args.max_velocity, self.args.min_velocity)
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
        np.add(width+1, self.dx, out=self.dx, where=-self.dx > width/2)
        np.add(height+1, self.dy, out=self.dy, where=-self.dy > height/2)

        # return hypotenuse of corresponding elements i.e. distance
        return np.hypot(self.dx, self.dy)

    def calc_alignment(self, mask, count):
        # Compute the average velocity of local neighbours
        target = np.dot(mask, self.velocity)/count
        
        # Compute steering
        norm = np.sqrt((target*target).sum(axis=1)).reshape(n, 1)
        target = self.args.max_velocity * np.divide(target, norm, out=target, where=norm != 0)
        target -= self.velocity

        limit(target, self.args.max_acceleration)
        return target


    def calc_cohesion(self, mask, count):
        # Compute the gravity center of local neighbours
        target = np.dot(mask, self.position)/count

        # Compute direction toward the center
        np.subtract(target, self.position, out=target, where=target != [0,0])

        # Wrap around
        np.subtract(target[:,0], np.sign(target[:,0])*width, out=target[:,0], where=np.absolute(target[:,0]) > width/2)
        np.subtract(target[:,1], np.sign(target[:,1])*height, out=target[:,1], where=np.absolute(target[:,1]) > height/2)
        
        # Normalize the result
        norm = np.sqrt((target*target).sum(axis=1)).reshape(n, 1)
        target = self.args.max_velocity * np.divide(target, norm, out=target, where=norm != 0)
        
        # Compute the resulting steering
        target -= self.velocity

        limit(target, self.args.max_acceleration)
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
        target = self.args.max_velocity * np.divide(target, norm, out=target, where=norm != 0)
        
        # Compute the resulting steering
        target -= self.velocity

        limit(target, self.args.max_acceleration)
        return target

    def random_turn(self):
        if self.args.eta:
            angles = np.random.uniform(-self.args.eta/2,self.args.eta/2,n)
        else:
            angles = np.random.normal(0,self.args.random,n)
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
defaults.angle = 60.
defaults.max_velocity = 1.
defaults.min_velocity = 0.5
defaults.v = None
defaults.max_acceleration = 0.03
defaults.width = 640
defaults.height = 480
defaults.n = 500
defaults.random = 0.1
defaults.alignment_radius = 50
defaults.cohesion_radius = 50
defaults.separation_radius = 25
defaults.alignment = 1
defaults.cohesion = 1
defaults.separation = 1.5
defaults.frames = 1000
defaults.vfile = "birds.mp4"
defaults.fps = 40
defaults.limit_view = "alignment,cohesion"
defaults.eta = 0

# -----------------------------------------------------------------------------
if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", "-a", help="Boid field of Vision [deg], default="+str(defaults.angle), type=float, default=defaults.angle)
    parser.add_argument("--max_velocity", help="Maximum velocity for a boid, default="+str(defaults.max_velocity), type=float, default=defaults.max_velocity)
    parser.add_argument("--min_velocity", help="Minimum velocity for a boid, default="+str(defaults.min_velocity), type=float, default=defaults.min_velocity)
    parser.add_argument("-v", help="Velocity parameter: Short for --max_velocity VELOCITY --min_velocity VELOCITY", type=float)
    parser.add_argument("--max_acceleration", help="Maximum acceleration per effect", type=float, default=defaults.max_acceleration)
    parser.add_argument("--width", help="Range for x-coordinate, default="+str(defaults.width), type=int, default=defaults.width)
    parser.add_argument("--height", help="Range for y-coordinate, default="+str(defaults.height),type=int, default=defaults.height)
    parser.add_argument("--n", "-n", help="Number of boids, default="+str(defaults.n),type=int, default=defaults.n)
    parser.add_argument("--random", "-r", help="Scale factor of normal distribution of random turning angles, default="+str(defaults.random), type=float, default=defaults.random)
    parser.add_argument("--eta", help="Set for uniformly distributed angles instead of normal distribution, generates Angles -eta/2 < delta Theta < eta/2", type=float, default=defaults.eta)
    parser.add_argument("--alignment_radius", help="Radius for boids considered in alignment, default="+str(defaults.alignment_radius), type=int, default=defaults.alignment_radius)
    parser.add_argument("--cohesion_radius", help="Radius for boids considered in cohesion, default="+str(defaults.cohesion_radius), type=int, default=defaults.cohesion_radius)
    parser.add_argument("--separation_radius", help="Radius for boids considered in separation, default="+str(defaults.separation_radius), type=int, default=defaults.separation_radius)
    parser.add_argument("--alignment", help="Weight of alignment in acceleration sum, default="+str(defaults.alignment), type=float, default=defaults.alignment)
    parser.add_argument("--cohesion", help="Weight of cohesion in acceleration sum, default="+str(defaults.cohesion), type=float, default=defaults.alignment)
    parser.add_argument("--separation", help="Weight of separation in acceleration sum, default="+str(defaults.separation), type=float, default=defaults.separation)
    parser.add_argument("--limit_view", help="Which effects are affected by viewing angle limitation, default="+defaults.limit_view, type=str, default=defaults.limit_view)
    parser.add_argument("--export", "-e", help="Export video file and exit", action="store_true")
    parser.add_argument("--frames", help="Number of frames for export to video or parameter record file, default="+str(defaults.frames), type=int, default=defaults.frames)
    parser.add_argument("--vfile", help="Out-File for video export, default="+str(defaults.vfile), type=str, default=defaults.vfile)
    parser.add_argument("--fps", help="Set Video Framerate for export, default="+str(defaults.fps), type=int, default=defaults.fps)
    parser.add_argument("--speed", help="Modify speed: Scale max_velocity, min_velocity, max_acceleration, alignment_radius, cohesion_radius, separation_radius at once.", type=float)
    parser.add_argument("--scale", help="Scale field size.", type=float)
    parser.add_argument("-s", help="equals --speed S --scale S", type=float)
    parser.add_argument("--out", "-o", help="Specify output File for recording Parameters, Filetype: .npz", type=str)
    parser.add_argument("--batch", "-b", help="Batch mode: no live display, no video export", action="store_true")




    args = parser.parse_args()

    if args.v:
        args.max_velocity = args.v
        args.min_velocity = args.v

    if args.s:
        args.speed = args.s
        args.scale = args.s

    if args.speed:
        args.max_velocity *= args.speed
        args.min_velocity *= args.speed
        args.max_acceleration *= args.speed
        args.alignment_radius *= args.speed
        args.cohesion_radius *= args.speed
        args.separation_radius *= args.speed

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



