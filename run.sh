
for MODEL in "inception_v3" "squeezenet" "vgg_11" "vgg_13" "alexnet" "resnet18" "resnet50"
do
    for OPT_TYPE in "dp_merge_parallel_transform", "dp_merge_parallel"
    do
    	python3 ./examples/transform_conv_optimize.py -m $MODEL -t $OPT_TYPE
    done
done
