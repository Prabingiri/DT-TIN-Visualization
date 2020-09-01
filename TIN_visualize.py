import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import math
import numpy as np

class visualize:

    def __init__(self, dataset):
        # self.cd_data = dict()
        # pass
        self.dataset = dataset


    def lon_lat_to_XYZ(self, lon: float, lat: float):
        # Convert angluar to cartesian coordiantes
        r = 6371  # https://en.wikipedia.org/wiki/Earth_radius
        theta = math.pi / 2 - math.radians(lat)
        phi = math.radians(lon)
        return r * math.sin(theta) * math.sin(phi), r * math.sin(theta) * math.cos(phi)

    def val_and_lon_lat(self, val_idx:int):
        loc_time_series = dict()
        rawdata = open('path/for/dataset' + self.dataset + '.txt', 'r')


        z = []
        for dpoints in rawdata:
            dpoints = dpoints.split(',')
            lon, lat = self.lon_lat_to_XYZ(float(dpoints[0]), float(dpoints[1]))
            time_series = float(dpoints[val_idx])
            z.append(time_series)
            loc_time_series[(lon, lat)] = time_series
        points = np.array(list(loc_time_series.keys()))
        lon = points[:][:, 0]
        lat = points[:][:, 1]
        return lon, lat, z

    def plotting(self, lon, lat, z_instance1, z_instance2, instance1, instance2, dataset):
        # Create the Triangulation; no triangles so Delaunay triangulation created.
        triang = mtri.Triangulation(lon, lat)
        fig = plt.figure(figsize=plt.figaspect(0.5))
        tri = mtri.Triangulation(lon, lat)
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot_trisurf(lon, lat, z_instance2, triangles=tri.triangles, cmap=plt.cm.Spectral)
        ax.set_title("data for the instance"+str(instance1))
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        ax.set_zlabel('values')



        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_trisurf(triang, z_instance1, cmap=plt.cm.Spectral)
        ax.set_title("Data in instance"+str(instance2))
        ax.set_xlabel('longitude')
        ax.set_ylabel('latitude')
        ax.set_zlabel('values')
        # ax.legend(z)
        # plt.colorbar(ax=z)
        #
        m = plt.cm.ScalarMappable(cmap=plt.cm.Spectral)
        m.set_array(z_instance1)
        # plt.colorbar(m)
        # plt.show()
        # path to store image
        return plt.savefig('/TINinstance/3D{0}.png'.format(datset.lower()))





