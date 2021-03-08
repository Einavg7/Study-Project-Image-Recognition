### Trackable object class
###### As the name suggests, it tracks the object/person and stores the centroid in a list. First, this part of code initializes the list of centroids and then stores the object/person ID.

```python
class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
```
###### In this part, the code keeps track of the count in each of the direction we worked on i.e. entrance as upc,  exit as downc, hall as hallc and office as officec.
```python
		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.upc = False
		self.downc = False
		self.hallc = False
		self.officec = False
```
