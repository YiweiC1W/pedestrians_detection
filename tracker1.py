import math


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points_list=[]
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 1
        self.now_frame=0
    def new_frame(self):
        self.center_points_list.append(self.center_points.copy())
        self.center_points={}
        self.now_frame+=1

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            #每次传入新的x,y,w,h
            x, y, w, h = rect
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)

            #找到已经存在的点
            object_exist = False
            for object_id, pt in self.center_points.items():
                distance = math.hypot(cx - pt[0], cy - pt[1])

                #如果distance在30pixels内则为同一object
                if distance < 40:
                    self.center_points[object_id] = (cx,cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, object_id])
                    object_exist = True
                    break

            #如果distance大于30pixels，object变换
            if not object_exist:
                self.center_points[self.id_count] = (cx,cy)
                objects_bbs_ids.append ([x,y,w,h,self.id_count])
                self.id_count += 1

        self.update_center(objects_bbs_ids)
        return objects_bbs_ids


    def update_center(self,objects_bbs_ids):
        #通过center_point更新object_id
        new_center_points= {}
        for ori_obj_bbs_id in objects_bbs_ids:
            _ ,_ ,_ ,_ , object_id = ori_obj_bbs_id
            moving_center = self.center_points[object_id]
            new_center_points[object_id] = moving_center

        # 完成迭代
        self.center_points = new_center_points.copy()










