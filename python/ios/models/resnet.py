from .common import *

def resnet_front(v):
    block = Block(v.node, None, [], None)
    v = conv2d(block, [[v]], out_channels=64, kernel=(7, 7), stride=(2, 2), padding=(3, 3), act='relu')
    v = pool2d(block, [[v]], pool_type='max', kernel=(3, 3), stride=(2, 2), padding=(1, 1), is_exit=True)
    return v, block, 64

def basic_block(block, v, channels, stride, downsample, is_exit):
    skip = v
    if downsample is not None:
        skip = downsample
    v = conv2d(block, [[v]], out_channels=channels, kernel=(3, 3), stride=stride, padding=(1, 1), act='relu')
    v = conv2d(block, [[v]], out_channels=channels, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='identity')
    v = addition(block, [[v, skip]], is_exit=is_exit)
    return v

def bottleneck(block, v, channels, stride, downsample, is_exit):
    skip = v
    if downsample is not None:
        skip = downsample
    v = conv2d(block, [[v]], out_channels=channels, kernel=(1, 1), stride=stride, padding=(0, 0), act='relu')
    v = conv2d(block, [[v]], out_channels=channels, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu')
    v = conv2d(block, [[v]], out_channels=channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0), act='identity')
    v = addition(block, [[v, skip]])
    v = activation(block, [[v]], act_type='relu', inplace=True, is_exit=is_exit)
    return v

def resnet_block(v, block_func, expansion, channels, layers, in_channels, stride):
    block = Block(v.node, None, [], None)
    if max(stride) != 1 or in_channels != channels * expansion:
        downsample = conv2d(block, [[v]], out_channels=channels * expansion, kernel=(1, 1), stride=stride, padding=(0, 0), act='relu')
    else:
        downsample = None
    v = block_func(block, v, channels, stride, downsample, is_exit=False)
    for t in range(1, layers):
        v = block_func(block, v, channels, stride=(1, 1), downsample=None, is_exit=(t == layers-1))
    return v, block, channels



def resnet_front_layout(v, layout=[DEFAULT_LAYOUT]*2, prev_layout=DEFAULT_LAYOUT):
    block = Block(v.node, None, [], None)
    conv_in_layout= layout[0]
    conv_out_layout = layout[1]
    if prev_layout != conv_in_layout:
        v = transform(block, [[v]], dst_layout = conv_in_layout)
    v = conv2d(block, [[v]], out_channels=64, kernel=(7, 7), stride=(2, 2), padding=(3, 3), act='relu', layout=conv_out_layout)
    v = pool2d(block, [[v]], pool_type='max', kernel=(3, 3), stride=(2, 2), padding=(1, 1), is_exit=True)
    return v, block, 64, conv_out_layout

def basic_block_layout(block, v, channels, stride, downsample, is_exit, layout=[[DEFAULT_LAYOUT]*2]*2, prev_layout=DEFAULT_LAYOUT):
    skip = v
    if downsample is not None:
        skip = downsample
    conv_in_layout= layout[0][0]
    conv_out_layout = layout[0][1]
    if prev_layout != conv_in_layout:
        v = transform(block, [[v]], dst_layout = conv_in_layout)
    v = conv2d(block, [[v]], out_channels=channels, kernel=(3, 3), stride=stride, padding=(1, 1), act='relu', layout=conv_out_layout)
    prev_layout = conv_out_layout
    conv_in_layout= layout[1][0]
    conv_out_layout = layout[1][1]
    if prev_layout != conv_in_layout:
        v = transform(block, [[v]], dst_layout = conv_in_layout)
    v = conv2d(block, [[v]], out_channels=channels, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='identity', layout=conv_out_layout)
    v = addition(block, [[v, skip]], is_exit=is_exit)
    return v, conv_out_layout

def bottleneck_layout(block, v, channels, stride, downsample, is_exit, layout=[[DEFAULT_LAYOUT]*2]*3, prev_layout=DEFAULT_LAYOUT):
    skip = v
    if downsample is not None:
        skip = downsample
    conv_in_layout= layout[0][0]
    conv_out_layout = layout[0][1]
    if prev_layout != conv_in_layout:
        v = transform(block, [[v]], dst_layout = conv_in_layout)
    v = conv2d(block, [[v]], out_channels=channels, kernel=(1, 1), stride=stride, padding=(0, 0), act='relu', layout=conv_out_layout)
    prev_layout = conv_out_layout
    conv_in_layout= layout[1][0]
    conv_out_layout = layout[1][1]
    if prev_layout != conv_in_layout:
        v = transform(block, [[v]], dst_layout = conv_in_layout)
    v = conv2d(block, [[v]], out_channels=channels, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout=conv_out_layout)
    prev_layout = conv_out_layout
    conv_in_layout= layout[2][0]
    conv_out_layout = layout[2][1]
    if prev_layout != conv_in_layout:
        v = transform(block, [[v]], dst_layout = conv_in_layout)
    v = conv2d(block, [[v]], out_channels=channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0), act='identity', layout=conv_out_layout)
    v = addition(block, [[v, skip]])
    v = activation(block, [[v]], act_type='relu', inplace=True, is_exit=is_exit)
    return v, conv_out_layout

def resnet_block_layout(v, block_func, expansion, channels, layers, in_channels, stride, layout=None, prev_layout=DEFAULT_LAYOUT):
    block = Block(v.node, None, [], None)
    if max(stride) != 1 or in_channels != channels * expansion:
        if layout is None:
            layout = [[DEFAULT_LAYOUT]*2] * (layers * 2 + 1)
        conv_in_layout= layout[0][0]
        conv_out_layout = layout[0][1]
        if prev_layout != conv_in_layout:
            v = transform(block, [[v]], dst_layout = conv_in_layout)
        downsample = conv2d(block, [[v]], out_channels=channels * expansion, kernel=(1, 1), stride=stride, padding=(0, 0), act='relu', layout=conv_out_layout)
        prev_layout = conv_out_layout
        remaining_layout = layout[1:]
    else:
        if layout is None:
            layout = [[DEFAULT_LAYOUT]*2] * (layers * 2)
        downsample = None
        remaining_layout = layout
    if block_func == basic_block_layout:
        layout_num = 2
    else:
        layout_num = 3
    v, prev_layout = block_func(block, v, channels, stride, downsample, is_exit=False, layout=remaining_layout[0:layout_num], prev_layout=prev_layout)
    for t in range(1, layers):
        v, prev_layout = block_func(block, v, channels, stride=(1, 1), downsample=None, is_exit=(t == layers-1), layout=remaining_layout[t*layout_num:(t+1)*layout_num], prev_layout=prev_layout)
    return v, block, channels, prev_layout

def resnet18():
    reset_name()

    pv = placeholder(output_shape=(3, 224, 224), layout=DEFAULT_LAYOUT)
    v, block1, out_channels = resnet_front(pv)
    v, block2, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=64,  layers=2, in_channels=out_channels, stride=(1, 1))
    v, block3, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=128, layers=2, in_channels=out_channels, stride=(2, 2))
    v, block4, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=256, layers=2, in_channels=out_channels, stride=(2, 2))
    v, block5, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=512, layers=2, in_channels=out_channels, stride=(2, 2))

    graph = Graph("resnet18", pv.node, [block1, block2, block3, block4, block5])
    graph.init_weights()
    graph.sequential_schedule()
    return graph

def resnet34():
    reset_name()

    pv = placeholder(output_shape=(3, 224, 224))
    v, block1, out_channels = resnet_front(pv)
    v, block2, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=64,  layers=3, in_channels=out_channels, stride=(1, 1))
    v, block3, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=128, layers=4, in_channels=out_channels, stride=(2, 2))
    v, block4, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=256, layers=6, in_channels=out_channels, stride=(2, 2))
    v, block5, out_channels = resnet_block(v, block_func=basic_block, expansion=1, channels=512, layers=3, in_channels=out_channels, stride=(2, 2))

    graph = Graph("resnet34", pv.node, [block1, block2, block3, block4, block5])
    graph.init_weights()
    graph.sequential_schedule()
    return graph

def resnet50():
    reset_name()

    pv = placeholder(output_shape=(3, 224, 224))
    v, block1, out_channels = resnet_front(pv)
    v, block2, out_channels = resnet_block(v, block_func=bottleneck, expansion=4, channels=64,  layers=3, in_channels=out_channels, stride=(1, 1))
    v, block3, out_channels = resnet_block(v, block_func=bottleneck, expansion=4, channels=128, layers=4, in_channels=out_channels, stride=(2, 2))
    v, block4, out_channels = resnet_block(v, block_func=bottleneck, expansion=4, channels=256, layers=6, in_channels=out_channels, stride=(2, 2))
    v, block5, out_channels = resnet_block(v, block_func=bottleneck, expansion=4, channels=512, layers=3, in_channels=out_channels, stride=(2, 2))

    graph = Graph("resnet50", pv.node, [block1, block2, block3, block4, block5])
    graph.init_weights()
    graph.sequential_schedule()
    return graph

def resnet18_opt_layout(layouts):
    reset_name()
    pv = placeholder(output_shape=(3, 224, 224), layout=layouts[0][0])
    v, block1, out_channels, prev_layout = resnet_front_layout(pv, layouts[0], prev_layout=layouts[0][0])
    v, block2, out_channels, prev_layout = resnet_block_layout(v, block_func=basic_block_layout, expansion=1, channels=64,  layers=2, in_channels=out_channels, stride=(1, 1), layout=layouts[1:5], prev_layout=prev_layout)
    v, block3, out_channels, prev_layout = resnet_block_layout(v, block_func=basic_block_layout, expansion=1, channels=128, layers=2, in_channels=out_channels, stride=(2, 2), layout=layouts[5:5+5], prev_layout=prev_layout)
    v, block4, out_channels, prev_layout = resnet_block_layout(v, block_func=basic_block_layout, expansion=1, channels=256, layers=2, in_channels=out_channels, stride=(2, 2), layout=layouts[10:10+5], prev_layout=prev_layout)
    v, block5, out_channels, prev_layout = resnet_block_layout(v, block_func=basic_block_layout, expansion=1, channels=512, layers=2, in_channels=out_channels, stride=(2, 2), layout=layouts[15:15+5], prev_layout=prev_layout)

    graph = Graph("resnet18", pv.node, [block1, block2, block3, block4, block5])
    graph.init_weights()
    graph.sequential_schedule()
    return graph