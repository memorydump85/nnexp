docker-run: docker-image
	docker run --rm -it --gpus=all nnexp-paired_feature_cnn python mnist.py

docker-run-bash: docker-image
	docker run --rm -it --gpus=all nnexp-paired_feature_cnn bash

docker-image:
	docker build ./ -t nnexp-paired_feature_cnn
