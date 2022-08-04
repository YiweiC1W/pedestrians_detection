# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
class CentroidTracker():
	def __init__(self, maxDisappeared=50):

		#分配ID
		self.nextObjectID = 0
		#对象字典
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		#达到最大连续帧数删除
		self.maxDisappeared = maxDisappeared

	#增加新的对象
	def register(self, bbx):
		self.objects[self.nextObjectID] = bbx
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	#移除离开的对象
	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]

	def get_dist(self,box_a,box_b):

		center_a = ((box_a[2]+box_a[0])//2,(box_a[3]+box_a[1])//2)
		center_b = ((box_b[2]+box_b[0])//2,(box_b[3]+box_b[1])//2)

		return ((center_b[0]-center_a[0])**2 + 	(center_b[1]-center_a[1])**2)**0.5


	def update1(self,rects):
		# 没有匹配的情况
		if len(rects) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			return self.objects
		# load data
		inputbbx = np.array(rects)

		if len(self.objects) == 0:
			for i in range(0, len(inputbbx)):
				self.register(inputbbx[i])
		else:
			objectIDs = list(self.objects.keys())
			objectbbx = list(self.objects.values())
			Distance = dist.cdist(np.array(objectbbx), inputbbx)
			rows = Distance.min(axis=1).argsort()
			cols = Distance.argmin(axis=1)[rows]

			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):

				if row in usedRows or col in usedCols:
					continue

				objectID = objectIDs[row]

				dis=self.get_dist(inputbbx[col],self.objects[objectID])
				if (dis>40):
					print(dis)
					continue
				self.objects[objectID] = inputbbx[col]
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)

			# 未处理的质心
			unusedRows = set(range(0, Distance.shape[0])).difference(usedRows)
			unusedCols = set(range(0, Distance.shape[1])).difference(usedCols)

			if Distance.shape[0] >= Distance.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:

					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			else:
				for col in unusedCols:
					self.register(inputbbx[col])

		# return the set of trackable objects
		appeared=OrderedDict()
		for item in self.disappeared.items():
			if item[1]==0:
				appeared[item[0]]=self.objects[item[0]]
		return appeared



	#质心坐标
	def update(self, rects):
		if len(rects) == 0:

			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			return self.objects

		# initialize an array of input centroids for the current frame
		inputbbx = np.zeros((len(rects), 4), dtype="int")
		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):

			Cx = int((startX + endX) / 2.0)
			Cy = int((startY + endY) / 2.0)
			inputbbx[i] = (startX, startY, endX, endY)
			bbx_center = (Cx,Cy)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputbbx)):
				self.register(inputbbx[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			#获取目标id
			objectIDs = list(self.objects.keys())
			objectbbx = list(self.objects.values())

			#计算距离排序
			Distance = dist.cdist(np.array(objectbbx), inputbbx)
			rows = Distance.min(axis=1).argsort()
			cols = Distance.argmin(axis=1)[rows]


			usedRows = set()
			usedCols = set()
			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):

				if row in usedRows or col in usedCols:
					continue

				objectID = objectIDs[row]
				self.objects[objectID] = inputbbx[col]
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)

			#未处理的质心
			unusedRows = set(range(0, Distance.shape[0])).difference(usedRows)
			unusedCols = set(range(0, Distance.shape[1])).difference(usedCols)

			if Distance.shape[0] >= Distance.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:

					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			else:
				for col in unusedCols:
					self.register(inputbbx[col])

		# return the set of trackable objects
		return self.objects