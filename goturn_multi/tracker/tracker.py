# Date: Friday 02 June 2017 05:04:00 PM IST
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: Basic regressor function implemented

from __future__ import print_function
from ..helper.image_proc import cropPadImage
from ..helper.BoundingBox import BoundingBox


class tracker:
    """tracker class"""

    def __init__(self, show_intermediate_output, logger):
        """TODO: to be defined. """
        self.show_intermediate_output = show_intermediate_output
        self.logger = logger

    def init(self, image_curr, bboxes_gt, objRegressor):
        """ initializing the first frame in the video
        """
        self.image_prev = image_curr
        self.bbox_prev_tight = bboxes_gt
        self.bbox_curr_prior_tight = bboxes_gt
        # objRegressor.init()

    def track(self, image_curr, objRegressor):
        """TODO: Docstring for tracker.
        :returns: TODO

        """
        target_batch = []
        search_batch = []
        search_location_batch = []
        edge_spacing_x_batch = []
        edge_spacing_y_batch = []
        for bbox in self.bbox_prev_tight:
            target_pad, _, _,  _ = cropPadImage(bbox, self.image_prev)
            target_batch.append(target_pad)
            cur_search_region,search_location,edge_spacing_x,edge_spacing_y=cropPadImage(bbox, image_curr)
            search_batch.append(cur_search_region)
            search_location_batch.append(search_location)
            edge_spacing_x_batch.append(edge_spacing_x)
            edge_spacing_y_batch.append(edge_spacing_y)

        bbox_estimates = objRegressor.regress(search_batch, target_batch)
        cur_bboxes = []
        for idx in range(len(bbox_estimates)):
            bbox_estimate = BoundingBox(bbox_estimates[idx, 0],
                    bbox_estimates[idx, 1], bbox_estimates[idx, 2],
                    bbox_estimates[idx, 3])

            # Inplace correction of bounding box
            bbox_estimate.unscale(search_batch[idx])
            bbox_estimate.uncenter(image_curr, search_location_batch[idx],
                    edge_spacing_x_batch[idx], edge_spacing_y_batch[idx])
            cur_bboxes.append(bbox_estimate)

        self.image_prev = image_curr
        self.bbox_prev_tight = cur_bboxes
        self.bbox_curr_prior_tight = cur_bboxes

        return cur_bboxes
