from scipy.spatial import Delaunay
import math
import numpy as np
from typing import Any, List, Dict, Union, Optional
from scipy.spatial.distance import directed_hausdorff
import pickle
# import compressionmethods
# import distance_metr

class distance_metrics:
    def __init__(self, dataset):

        self.dataset = dataset
        self.raw_volume = dict()
        self.whole_vol_res = dict()
        self.res_each_cluster = dict()
        self.loc_ts = dict()


    def preprocessing(self):

        for ts in self.rawdata:
            ts = ts.split(',')
            lon, lat = self.lon_lat_to_XYZ(float(ts[0]), float(ts[1]))
            time_s = [float(each_one) for each_one in ts[2:]]
            self.loc_ts[(lon, lat)] = time_s
        return self.loc_ts




    def lon_lat_to_XYZ(self, lon: float, lat: float):
        # Convert angluar to cartesian coordiantes
        r = 6371  # https://en.wikipedia.org/wiki/Earth_radius
        theta = math.pi / 2 - math.radians(lat)
        phi = math.radians(lon)
        return r * math.sin(theta) * math.sin(phi), r * math.sin(theta) * math.cos(phi)

    def calc_Volume(self, p1, p2, p3):
        # print("calc_vol")
        base_area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        h = 1 / 3 * (p1[2] + p2[2] + p3[2])
        volume = base_area * h
        return volume

    def Volumetric_Difference(self, instance1, instance2):
        # raw_volume=dict()
        loc_time_series = dict()
        rawdata = open('/path/to/dataset/'+self.dataset+ '.txt', 'r')


        for dpoints in rawdata:
            dpoints = dpoints.split(',')
            lon, lat = self.lon_lat_to_XYZ(float(dpoints[0]), float(dpoints[1]))
            time_series = [float(each_one) for each_one in dpoints[2:]]
            loc_time_series[(lon, lat)] = time_series
        points = np.array(list(loc_time_series.keys()))
        tris = Delaunay(points)
        # print(tris)
        # print(len(tris.simplices))

        sum_vol_instance_1 = 0
        sum_vol_instance_2 = 0
        for each_tri in points[tris.simplices]:
            p1, p2, p3 = tuple(each_tri[0]), tuple(each_tri[1]), tuple(each_tri[2])
            # print(p1, p2, p3)
            ts1, ts2, ts3 = loc_time_series[p1], loc_time_series[p2], loc_time_series[p3]
            # print(ts1, ts2, ts3)
            # print(len(ts1[0]))
            # print(len(points))
            # for instance 1

            first_instance = self.calc_Volume(list(p1) + [ts1[instance1]], list(p2) + [ts2[instance1]],
                              list(p3) + [ts3[instance1]])
            # print(first_instance)
            sum_vol_instance_1 += first_instance
            # print(sum_vol_instance_1)

            second_instance = self.calc_Volume(list(p1) + [ts1[instance2]], list(p2) + [ts2[instance2]],
                              list(p3) + [ts3[instance2]])
            # print(first_instance, second_instance)
            sum_vol_instance_2 += second_instance
            # print(sum_vol_instance_1, sum_vol_instance_2)




        # print(sum_vol_instance_1, sum_vol_instance_2)
        vol_diff = abs(sum_vol_instance_1-sum_vol_instance_2)
        # print(vol_diff)
        rawdata.close()
        return vol_diff




    def Hausdarff_distance(self, instance1, instance2):
        data = open('path/to/dataset/' + self.dataset + '.txt', 'r')
        loc_ts = dict()
        for ts in data:
            ts = ts.split(',')
            lon, lat = self.lon_lat_to_XYZ(float(ts[0]), float(ts[1]))
            time_s = [float(each_one) for each_one in ts[2:]]
            loc_ts[(lon, lat)] = time_s
        points = np.array(list(loc_ts.keys()))
        tris = Delaunay(points)
            # print(tris)
            # print(len(tris.simplices))

        sum_vol_instance_1 = 0
        sum_vol_instance_2 = 0
        h_d = []
        for each_tri in points[tris.simplices]:

            p1, p2, p3 = tuple(each_tri[0]), tuple(each_tri[1]), tuple(each_tri[2])
            # print(p1, p2, p3)
            ts1, ts2, ts3 = loc_ts[p1], loc_ts[p2], loc_ts[p3]

            instanc1_TIN = [(p1[0], p1[1], ts1[instance1]), (p2[0], p2[1], ts2[instance1]), (p3[0], p3[1], ts3[instance1])]
            instance2_TIN = [(p1[0], p1[1], ts1[instance2]), (p2[0], p2[1], ts2[instance2]), (p3[0], p3[1], ts3[instance2])]
            # print(h_d)
            h_d.append(max(directed_hausdorff(instanc1_TIN, instance2_TIN)[0], directed_hausdorff(instanc1_TIN, instance2_TIN)[0]))

        return h_d