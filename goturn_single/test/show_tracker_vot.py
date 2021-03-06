# Date: Friday 02 June 2017 05:50:20 PM IST
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: Test file for showing the tracker output
import sys
import argparse
#import setproctitle
from ..logger.logger import setup_logger
#from ..network.regressor import regressor
from ..network.regressor import AlexNet
from ..loader.loader_vot import loader_vot
#from ..loader.loader_vot import loader_bit
from ..tracker.tracker import tracker
from ..tracker.tracker_manager import tracker_manager
import torch 
#setproctitle.setproctitle('SHOW_TRACKER_VOT')
logger = setup_logger(logfile=None)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="Path to the prototxt")
ap.add_argument("-m", "--model", required=True, help="Path to the model")
ap.add_argument("-v", "--input", required=True, help="Path to the vot directory")
ap.add_argument("-g", "--gpuID", required=True, help="gpu to use")
args = vars(ap.parse_args())

do_train = False
#objRegressor = regressor(args['prototxt'], args['model'], args['gpuID'], 1, do_train, logger)
objRegressor = AlexNet()
alex_dict = objRegressor.state_dict()
pre = torch.load(args['model'])
new = {k.replace('-','_'):v for k, v in pre.items() if '_p' not in k}
alex_dict.update(new)
objRegressor.load_state_dict(alex_dict)
#objRegressor.double()
objRegressor.eval()

objTracker = tracker(False, logger)  # Currently no idea why this class is needed, eventually we shall figure it out
objLoaderVot = loader_vot(args['input'], logger)
videos = objLoaderVot.get_videos()
objTrackerVis = tracker_manager(videos, objRegressor, objTracker, logger)
objTrackerVis.trackAll(0, 1)
