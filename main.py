from TIN_visualize import visualize
from distance_metrics import distance_metrics
# from app.visualization import visualize
def visualization(instance1, instance2, dataset):
    v = visualize(dataset)
    lon, lat, zinstance1 = v.val_and_lon_lat(val_idx=int(instance1) + 2)
    lon, lat, zinstance2 = v.val_and_lon_lat(val_idx=int(instance2) + 2)
    v.plotting(lon=lon, lat=lat, z_instance1=zinstance1, z_instance2=zinstance2, instance1=instance1, instance2=instance2, dataset=dataset)



if __name__ == '__main__':
    time_instance1 = 1
    time_instance2 = 250
    dataset = 'cluster1'
    visualization(time_instance1, time_instance2, dataset)
    d_metric = distance_metrics(dataset)
    volume_distance = d_metric.Volumetric_Difference(time_instance1, time_instance2)
    print(volume_distance)



