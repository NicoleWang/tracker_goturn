# Date: Wednesday 07 June 2017 11:28:11 AM IST
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: tracker manager

import cv2
import os
import time

opencv_version = cv2.__version__.split('.')[0]
colors = {'1':[(0,139,0),(0,255,127)], '2':[(26,26,139),(64,64,255)]}

class tracker_manager:

    """Docstring for tracker_manager. """

    def __init__(self, videos, regressor, tracker, logger):
        """This is

        :videos: list of video frames and annotations
        :regressor: regressor object
        :tracker: tracker object
        :logger: logger object
        :returns: list of video sub directories
        """

        self.videos = videos
        self.regressor = regressor
        self.tracker = tracker
        self.logger = logger

    def trackAll(self, start_video_num, pause_val):
        """Track the objects in the video
        """

        videos = self.videos
        objRegressor = self.regressor
        objTracker = self.tracker

        video_keys = videos.keys()
        for i in range(start_video_num, len(videos)):
            video_frames = videos[video_keys[i]][0]
            annot_frames = videos[video_keys[i]][1]

            num_frames = min(len(video_frames), len(annot_frames))

            # Get the first frame of this video with the intial ground-truth bounding box
            frame_0 = video_frames[0]
            bbox_0 = annot_frames[0] ##attention: multiple bboxes
            sMatImage = cv2.imread(frame_0)
            bboxes_init = []
            for bb in bbox_0:
                key = bb.keys()[0]
                bboxes_init.append(bb[key])
                print(key)
                bb[key].print_info()
            objTracker.init(sMatImage, bboxes_init, objRegressor)
            for i in xrange(1, num_frames):
                frame = video_frames[i]
                sMatImage = cv2.imread(frame)
                sMatImageDraw = sMatImage.copy()
                bboxes = annot_frames[i]
                subdir = frame.split('/')[-2]
                imname = frame.split('/')[-1]
                save_path = os.path.join("./show",imname)
                #print("gt")
                #print(bbox.x1, bbox.y1, bbox.x2, bbox.y2)
                index_dict = dict()
                if opencv_version != '2':
                    for idx, bb in enumerate(bboxes):
                        key = bb.keys()[0]
                        index_dict[key] = idx
                        bbox = bb[key]
                        color = colors[key][0]
                        cv2.rectangle(sMatImageDraw, (int(bbox.x1),
                            int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), color, 2)
                #else:
                #    sMatImageDraw = cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 255, 255), 2)
                start = time.time()
                bboxes = objTracker.track(sMatImage, objRegressor)
                end = time.time()
                #print("time: ",(end - start)*1000, " ms")
                if opencv_version != '2':
                    for idx, bbox in enumerate(bboxes):
                        for key in  index_dict.keys():
                            if index_dict[key] == idx:
                                color = colors[key][1]
                        cv2.rectangle(sMatImageDraw, (int(bbox.x1),
                            int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), color, 2)
                #else:
                #    sMatImageDraw = cv2.rectangle(sMatImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 0, 0), 2)

                cv2.imwrite(save_path, sMatImageDraw)
                #cv2.imshow('Results', sMatImageDraw)
                #cv2.waitKey(10)
