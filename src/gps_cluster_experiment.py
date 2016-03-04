import numpy as np
from sklearn import mixture
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sys
sys.path.append('D:\\documents\\hdp\\src')
import gps_cluster

def parse_time(time_str):
    i = time_str.rfind(':')
    return datetime.datetime.strptime(time_str[:i], '%Y-%m-%d %H:%M')

df = pd.DataFrame.from_csv('J:\\data\\beijing_center_09.txt', index_col=None)
df['datetime'] = df.time.map(parse_time)
df['time'] = df['datetime'].map(lambda dt: dt.time())
df['weekday'] = df['datetime'].map(lambda dt: dt.weekday())
morning_index = (df['time'] > datetime.time(6, 0, 0)) & (df['time'] < datetime.time(10, 0, 0))
afternoon_index = (df['time'] > datetime.time(16, 0, 0)) & (df['time'] < datetime.time(18, 0, 0))

index = morning_index
#index = index & (df['lat'] < 39.875) & (df['lon'] < 116.35)
lon = df.lon[index].values
lat = df.lat[index].values
hours = df.time[index].map(lambda t: t.hour).values
data = np.stack([lon, lat], axis=1)

hours = df.time.map(lambda t: t.hour).values
data = np.stack([df.lon, df.lat], axis=1)

def datetime_to_group(dt):
    i = dt.weekday()#0 if dt.weekday() < 5 else 1
    k = 0 if dt.minute < 30 else 1
    return i * 48 + dt.hour * 2 + k


def group_to_datetime(g):
    day = g / 48
    h = (g % 48) / 2
    m = (g % 48) % 2
    return day, h, m

from scipy import stats

xmin = lon.min()
xmax = lon.max()
ymin = lat.min()
ymax = lat.max()



X, Y = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([lon, lat])
kernel = stats.gaussian_kde(values, 0.08)
Z = np.reshape(kernel(positions).T, X.shape)


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
           extent=[xmin, xmax, ymin, ymax])
#ax.plot(gps[:,0], gps[:,1], 'k.', markersize=1)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.show()

n_components=10
covariance_type ='spherical'# 'full'#'diag'
gmm = mixture.GMM(n_components=n_components, covariance_type=covariance_type, min_covar=0.00001, n_iter=1000)
#gmm._set_covars(np.ones((n_components, 2)) * 0.05)
gmm.fit(data)


import itertools
import matplotlib as mpl
def plot_mixture(gmm, ax, scale=1.0):
    color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
    for n, color in zip(range(gmm.n_components), color_iter):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= scale
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


ax = plt.subplot(111)
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
           extent=[xmin, xmax, ymin, ymax])
plot_mixture(gmm, ax, 10.0)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.show()


from sklearn import cluster
n_components= 5
gmm = mixture.GMM(n_components=n_components, min_covar=0.00000001)
#y_gmm = gmm.fit_predict(data)
km = cluster.KMeans(n_components)
y_km = km.fit_predict(data)

ax = plt.subplot(211)
color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'k', 'y', 'w'])
for i, c in zip(range(km.n_clusters), color_iter):
    ix = (y_gmm == i)
    ax.scatter(data[ix, 0], data[ix, 1], c=c)
ax = plt.subplot(212)
color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'k', 'y', 'w'])
for i, c in zip(range(km.n_clusters), color_iter):
    ix = (y_km == i)
    ax.scatter(data[ix, 0], data[ix, 1], c=c)
plt.show()


reload(gps_cluster)
#group_label = group_data(data, 10, 10)
group_label = df.datetime.map(datetime_to_group).values
#group_label = np.zeros(data.shape[0], dtype=np.int)
gmm = gps_cluster.GassianExpert(70, 2, 0.00001, 2)
local_weight = gmm.fit(data, group_label, 50)
print gmm.means_, gmm.weight, gmm.vars
plt.scatter(gmm.means_[:, 0], gmm.means_[:,1])
plt.show()

plt.scatter(gmm.means_[:, 0], gmm.means_[:,1])
plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
           extent=[xmin, xmax, ymin, ymax])
plt.show()

#plt.plot(local_weight)# / local_weight.sum(axis=1)[:,np.newaxis])
comp_weight = local_weight.sum(axis=1)
max_componet = comp_weight.argmax()
plt.plot(local_weight[:, 37])
plt.show()

def group_data(X, m, n):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_step, y_step = (x_max - x_min) / n, (y_max - y_min) / m
    label = np.zeros(X.shape[0], dtype=np.int)
    for i in range(X.shape[0]):
        ix = min(int((X[i, 0] - x_min) / x_step), n-1)
        iy = min(int((X[i, 1] - y_min) / y_step), m-1)
        label[i] = ix + iy * n
    return label

a, cnt = np.unique(df.datetime.map(datetime_to_group), return_counts=True)
plt.bar(np.arange(cnt.shape[0]), cnt)


