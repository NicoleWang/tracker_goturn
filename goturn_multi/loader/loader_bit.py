# Date: Friday 02 June 2017 07:00:47 PM IST
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: loading VOT dataset

from __future__ import print_function
import os, json
from ..helper.BoundingBox import BoundingBox
from video import video
import glob


class loader_bit:
    """Helper functions for loading VOT data"""

    def __init__(self, vot_folder, logger):
        """Load all the videos in the vot_folder"""

        self.logger = logger
        self.vot_folder = vot_folder
        self.videos = {}
        self.annotations = {}
        if not os.path.isdir(vot_folder):
            logger.error('{} is not a valid directory'.format(vot_folder))

    def print_ann(self):
        for key, val in self.videos.iteritems():
            frames, anns = val
            for idx in range(len(frames)):
                ann = anns[idx]
                print(frames[idx])
                for k, v in ann.iteritems():
                    print(k)
                    v.print_info()

    def get_videos(self):
        """Docstring for get_videos.

        :returns: returns video frames in each sub folder of vot directory

        """
        
        logger = self.logger
        vot_folder = self.vot_folder
        sub_vot_dirs = self.find_subfolders(vot_folder)
        for vot_sub_dir in sub_vot_dirs:
            video_path = glob.glob(os.path.join(vot_folder, vot_sub_dir, '*.jpg'))
            objVid = video(video_path)
            list_of_frames = sorted(video_path)
            if not list_of_frames:
                logger.error('vot folders should contain only .jpg images')

            objVid.all_frames = list_of_frames
            bbox_gt_file = os.path.join(vot_folder, vot_sub_dir,
                    'groundtruth.json')
            with open(bbox_gt_file, 'r') as f:
                data = json.load(f)
                for i, line in enumerate(list_of_frames):
                    imname = line.split('/')[-1]
                    cur_ann = data[imname]
                    cur_bbs = []
                    for key, val in cur_ann.iteritems():
                        bbox = BoundingBox(*val)
                        bbox.frame_num = i
                        cur_bbs.append({key:bbox})
                        #cur_ann[key] = bbox
                    objVid.annotations.append(cur_bbs)
            self.videos[vot_sub_dir] = [objVid.all_frames, objVid.annotations]
            #self.print_ann()
        return self.videos
            

    def find_subfolders(self, vot_folder):
        """TODO: Docstring for find_subfolders.

        :vot_folder: directory for vot videos
        :returns: list of video sub directories
        """

        return [dir_name for dir_name in os.listdir(vot_folder) if os.path.isdir(os.path.join(vot_folder, dir_name))]
