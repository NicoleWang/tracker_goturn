DEPLOY_PROTO='./nets/tracker.prototxt'		 
CAFFE_MODEL='./nets/tracker.pt'		
TEST_DATA_PATH='./data/vot14/'		
#TEST_DATA_PATH='./data/multi'		

python -m goturn_single.test.show_tracker_vot \
	--p $DEPLOY_PROTO \
	--m $CAFFE_MODEL \
	--i $TEST_DATA_PATH \
	--g 0
