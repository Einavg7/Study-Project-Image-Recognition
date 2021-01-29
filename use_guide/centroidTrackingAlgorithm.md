### Import required libraries
###### Initial step is to import the necessray libraries for the centroid tracker algorithm. In the first step, we require pacakages and modules - distance, OrderedDict, Numpy and SciPy.
```python
# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
```
### Class Centroid Tracker
###### The class centroid tracker contains all the methods that would keep the algorithm flow smooth. The constructor is defined as *init* and the methods are *register*, *deregister* and *update*. Let's understand the purpose of each one by one.

1. ###### **__init__()**: This constructor accepts two parameters. First, the maximum number of consecutive frames a given object has to be lost/disappeared until we remove it from our tracker and second, for the maximum distance between two centriods in order to associate an object. Now, let's take a look a the class variables.
* ###### *nextobjectID* : The defined variable assigns unique ID's to each object. If an object/person with unique ID leaves the frame and is not detected using "maxDisappeared" frames, a new object ID would be assigned to the object/person.
* ###### *objects* : This variable works as a dictionary that stores the object/peron ID's as key and assignes the the centroid (x,y) coordinate of the same object/person as a value.
* ###### *disappeared* : To maintains the number of consecutive frames (position value) a particular object/person ID (key) has lost.
* ###### *maxDisappeared* : This variable keeps the track of the consecutive frames an  object/person is allowed to be marked as lost or disapppeared until we deregistered the object.
* ###### *maxDistance* : If the distance between the associated centriods excceds the maximum limit, we mark it as disappeared.
```python
class CentroidTracker:
	def __init__(self, maxDisappeared=50, maxDistance=50):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

		# store the maximum distance between centroids to associate
		# an object -- if the distance is larger than this maximum
		# distance we'll start to mark the object as "disappeared"
		self.maxDistance = maxDistance
```

2. ###### **register** : This method is responsible for adding new objects to our tracker.
* ###### First it accepts the centriod (x,y coordinates) and then adds it to the objects dictionary using tthe next available object ID.
* ###### Then the number of times an object has disappeared is initialized to 0 in the *disappeared* dictionary.
* ###### Finally, using *nextobjectID*, we increment a count each time a new object/person comes to the view. This is associated with the unique ID's.

```python
	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1
```
3. ###### **deregister** : To remove the lost/diappeared object/person ID's.
* ###### This method deletes the *objectID*  in both the *objects* and *disappeared* dictionaries.
```python
	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]
```
4. ###### **update** : This method accepts a list of bounding box rectangles, presumably from the object detector (SSD). the format of the *rects* parameter is a tuple with (startx, starty, endx, endy) structure. 
* ###### In update method, when no object/person ID's are no longer detected, *disappeared* count is increased by +1. 
* ###### In addition to that, we remove the object/person ID's after checking the maximum limit of the number of consecutive frames a given object has and mark as missing. 
*  ###### If there is nothing to update then we skip to *return*.

```python
	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects
```
* ###### A numpy arry is being initialized in order to store the centroids for each of the *rect*
* ###### Next step includes looping over bounding box rectangles and then it calculates the centroid position. the centroid values are stored in *inputcentroids* list.
* ###### 
```python
		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)
```
* ###### This chunk depicts the situation, when there are no trackalble object/person. Then it will register each new object/person ID. 
* ###### The *else* part, in general is needed to update any existing object/person with (x,y) coordinates which minimizes the Euclidean distance between them.
* ###### In order to track and maintain the correct obect/person ID's, we computed  the Eucliedean distance between all pairs of the *objectcentroids* and *inputcentroids*.
* ###### Besides, we also associate the object/person ID's to minimize the Euclidean distance. 
```python
		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]
```
* ###### This part of code explains how we used the distances of the object/person ID's for association.

``` python 
			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				if row in usedRows or col in usedCols:
					continue

				# if the distance between centroids is greater than
				# the maximum distance, do not associate the two
				# centroids to the same object
				if D[row, col] > self.maxDistance:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)
```
* ###### It is necessary that we determine the centroids that are not examined yet and store them in the unusedRows and unusedCols.
* ###### Finally, we check if any object that is lost/diasppeared.
```python
			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
```
* ###### The else part loops over the *unusedCols* indexes and we register each new centroid location (x,y coordinates) and then return the set of trackable objects to the calling method.

```python
			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects
```