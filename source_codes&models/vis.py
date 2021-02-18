import os
import csv
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import dates as mdate
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation; animation.writers.list()
import time
from datetime  import time
from shapely.geometry import Point, LineString, Polygon, MultiPoint
from geopandas import GeoDataFrame, read_file
from fiona.crs import from_epsg
import urllib
from holoviews import opts, dim

## visualizations for first video

projectDirectory="..."
inputCSV_25=os.path.join(projectDirectory, "results6.csv") #from 25.1 10:30-11:30


df1=pd.read_csv(inputCSV_25)
df1 = df1.groupby(['exit','entrance', 'hall', 'office']).time.unique().reset_index()
df1['time'] = df1['time'].str[0]
print(df1.head())


# plot animated count for each area at IFGI by time (first) 
Writer = animation.writers['ffmpeg']
writer = Writer(fps=2, metadata=dict(artist='Matplotlib'), bitrate=1800)

df1['time'] = pd.to_datetime(m['time'], format = "%H:%M:%S")
df1.sort_values('time', inplace = True)

cdate = df1['time']
ccase = df1['entrance']
cdeath = df1['exit']
chall=df1['hall']
coffice=df1['office']

fig = plt.figure(figsize=(20,10),facecolor='paleturquoise')
ax1 = fig.add_subplot(111)
#title = ax.text(0.5,0.85,"", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},transform=ax.transAxes, ha="center")
##tit=plt.title('People Counter')
##xl=plt.xlabel('Time')
##yl=plt.ylabel('Counts')
def animate(i):
    ax1.clear()
    ax1.set_facecolor('lightgrey')
    ax1.plot(cdate[:i], ccase[:i], label = 'Entrance',linewidth=2,marker="o")
    ax1.plot(cdate[:i], cdeath[:i], label = 'Exit',linewidth=2,marker="^")
    ax1.plot(cdate[:i], chall[:i], label = 'Hall',linewidth=2,marker="s")
    ax1.plot(cdate[:i], coffice[:i], label = 'Office',linewidth=2,marker="8")
    ax1.legend(loc = 'upper left')
    ax1.set_title('People Counter : IFGI (25.12.2020)',fontsize=20)
    ax1.set_xlabel('Time',fontsize=15)
    ax1.set_ylabel('Count',fontsize=15)
    ax1.set_xlim([cdate.iloc[0],
                  cdate.iloc[-1]])
    ax1.set_ylim([min(ccase.iloc[0], cdeath.iloc[0]),
                  max(ccase.iloc[-1], cdeath.iloc[-1])])
    ax1.grid(True)
    xlabels= ax1.xaxis.set_major_locator(mdate.MinuteLocator(interval = 3))
    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M:%S'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45 )
  

ani = animation.FuncAnimation(fig,animate, interval = 1000,save_count=58)
ani.save('F:/lin.mp4', writer=writer,dpi=100)
plt.show()

## plot trajectory for each ID

trajectory1=os.path.join(projectDirectory, "trajectory1.csv") #from 25.1 10:30-11:30

dftraj1=pd.read_csv(trajectory1)

fig = plt.figure()
ax2 = fig.add_subplot(111)
img = plt.imread("testhour1_Moment.jpg")
cmap = plt.get_cmap('tab20', 20)
ax2 = dftraj1.plot(kind='scatter', x = 'centroidx', y ='centroidy', c='ID',legend=False, cmap=cmap, vmin=0, vmax=67, s = 10,title='Trajectory of People in IFGI 25.1.2021')
ax2 = plt.imshow(img,  zorder=0, extent=[0, 400, 0, 300], alpha = 0.5, origin='lower')
ax2 = plt.gca().invert_yaxis()
filename2 = os.path.join(projectDirectory, "traj1.png")
# plt.savefig(filename2, dpi=600, format = "png")
plt.show()

## moving pandas trajectory analysis

# first we need trajectories to have a timestamp, thus grouping dataframe by id
# and making it a simple dict because I couldn't find how to do it all with
# pandas dataframes
dict_by_id = dict(iter(dftraj1.groupby('ID')))
new_dfs = []

# iterating through each trajectory and appending an arbitrary list of timestamps
for track_id in dict_by_id:

    # datetimes = pd.date_range("2020-01-01", periods=len(dict_by_id[track_id]), freq="S")
    # dict_by_id[track_id]['t'] = datetimes

    dates_objects = []
    for time in dict_by_id[track_id]['timestamp']:
        full_date = '18/09/19 ' + time
        date_time_obj = datetime.strptime(full_date, '%d/%m/%y %H:%M:%S')
        dates_objects.append(date_time_obj)
    
    dict_by_id[track_id]['t'] = dates_objects
    
    for key in dict_by_id[track_id]['ID']:
        key = int(key)

    dict_by_id[track_id]['speed'] = dict_by_id[track_id]['ID']
    # making each trajectory a separate pandas dataframe and appending it to a list of them
    new_pd = pd.DataFrame.from_dict(dict_by_id[track_id]).set_index('t')
    new_dfs.append(new_pd)
    
# concatenating all our new dataframes to a flat one
trajectories_df_with_time = pd.concat(new_dfs)


# now making it a geodataframe for movingpandas
trajectories_gdf = GeoDataFrame(
    trajectories_df_with_time.drop(['centroidx', 'centroidy'], axis=1),
    crs=from_epsg(31256),
    geometry=[Point(xy) for xy in zip(trajectories_df_with_time.centroidx, trajectories_df_with_time.centroidy)])

print('Type of id', type(trajectories_gdf.iloc[0]['ID']))

traj_collection = mpd.TrajectoryCollection(trajectories_gdf, 'ID')
# print(traj_collection)

# traj_collection.add_speed(overwrite=True)

traj_collection.plot(column="speed", linewidth=5, height=500, width=500, legend=True)
plt.gca().invert_yaxis()
plt.show()


trips = mpd.ObservationGapSplitter(traj_collection).split(gap=timedelta(minutes=5))
print("Extracted {} individual trips from {} tracks".format(len(trips), len(traj_collection)))

%%time
aggregator = mpd.TrajectoryCollectionAggregator(traj_collection, max_distance=100, min_distance=1, min_stop_duration=timedelta(minutes=0.2))

pts = aggregator.get_significant_points_gdf()
clusters = aggregator.get_clusters_gdf()
( pts.hvplot(geo=False, frame_width=400, frame_height=300, flip_yaxis=True) * 
  clusters.hvplot(geo=False, color='red', flip_yaxis=True) )

flows = aggregator.get_flows_gdf()
print(flows)

flows.hvplot(geo=False, hover_cols=['weight'], frame_width=400, frame_height=300, line_width=dim('weight')*0.3, alpha=0.5, color='#1f77b3', flip_yaxis=True) * 
  clusters.hvplot(geo=False, color='red', size='n', flip_yaxis=True) )
