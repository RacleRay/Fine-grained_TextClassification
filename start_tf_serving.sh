docker run -t --rm -p 8501:8501 -p 8500:8500 \
	-v "$(pwd)/serving/v1:/models/classifier" \
	-e MODEL_NAME=classifier \
	tensorflow/serving:1.15.0-gpu &> runing_log &
