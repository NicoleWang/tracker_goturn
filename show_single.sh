MODEL_PATH='./nets/tracker.pt'		
TEST_DATA_PATH='./data/vot14/'		
#TEST_DATA_PATH='./data/multi'		

python -m goturn_single.test.show_tracker_vot \
	--m $MODEL_PATH \
	--i $TEST_DATA_PATH \
	--g 0
