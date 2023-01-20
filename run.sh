cd build && cmake .. && make -j4 && cd ../python; python setup.py install && cd ..
for MODEL in "vgg_11" "vgg_13" "vgg_16" "vgg_19" "inception_v3" "squeezenet" "alexnet" "resnet18" "resnet34" "resnet50"
do   
    # for OPT_TYPE in "dp_merge_parallel_transform" "dp_merge_parallel"
    # do
    # 	python3 ./examples/transform_conv_optimize.py -m $MODEL -t $OPT_TYPE
    # done
    python3 ./examples/transform_layout.py -m $MODEL
    # python3 ./examples/conv_layout.py -m $MODEL
    # python3 ./examples/pool_layout.py -m $MODEL
    # python3 ./examples/elem_act_layout.py -m $MODEL
done
