from .common import *

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

layouts = {
    'A': [
        ["NHWC", "NCHW"],
        ["NCHW", "NCHW"],
        ["NCHW", "NHWC"], ["NHWC", "NCHW"],
        ["NCHW", "NCHW"], ["NCHW", "NHWC"],
        ["NHWC", "NCHW"], ["NCHW", "NCHW"]
    ],
    'B': [
        ["NHWC", "NHWC"], ["NHWC", "NHWC"],
        ["NCHW", "NCHW"], ["NCHW", "NCHW"],
        ["NCHW", "NHWC"], ["NHWC", "NHWC"],
        ["NHWC", "NCHW"], ["NCHW", "NCHW"],
        ["NCHW", "NCHW"], ["NCHW", "NCHW"],
    ]
}


def vgg_net(cfg, name):
    reset_name()

    pv = placeholder(output_shape=(3, 224, 224))
    block = Block(pv.node, None, [], None)

    v = pv
    for c in cfg:
        if c == 'M':
            v = pool2d(block, [[v]], pool_type='max', kernel=(2, 2), stride=(2, 2))
        else:
            v = conv2d(block, [[v]], c, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act="relu")
    v = pool2d(block, [[v]], pool_type='global_avg', is_exit=True)
    graph = Graph(name, pv.node, [block])
    graph.init_weights()
    return graph


def vgg_net_opt_layout(cfg, layout, name):
    print("layout", layout)
    reset_name()
    cnt = 0
    pv = placeholder(output_shape=(3, 224, 224), layout=layout[cnt][0])
    block = Block(pv.node, None, [], None)

    v = pv
    prev_layout = layout[cnt][0]
    for c in cfg:
        if c == 'M':
            v = pool2d(block, [[v]], pool_type='max', kernel=(2, 2), stride=(2, 2))
        else:
            conv_in_layout= layout[cnt][0]
            conv_out_layout = layout[cnt][1]
            # v = transform_conv2d(
            #     block, [[v]], c, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act="relu",
            #     conv_in_layout=conv_in_layout, conv_out_layout=conv_out_layout
            # )
            if prev_layout != conv_in_layout:
                v = transform(block, [[v]], dst_layout = conv_in_layout)
            v = conv2d(
                block, [[v]], c, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act="relu",
                layout=conv_out_layout
            )
            prev_layout = conv_out_layout
            cnt += 1
    v = pool2d(block, [[v]], pool_type='global_avg', is_exit=True)
    graph = Graph(name, pv.node, [block])
    graph.init_weights()
    return graph

def vgg_11():
    return vgg_net(cfgs['A'], 'vgg_11')


def vgg_13():
    return vgg_net(cfgs['B'], 'vgg_13')


def vgg_16():
    return vgg_net(cfgs['D'], 'vgg_16')


def vgg_19():
    return vgg_net(cfgs['E'], 'vgg_19')


def vgg_11_opt_layout():
    return vgg_net_opt_layout(cfgs["A"], layouts["A"], "vgg_11")

def vgg_13_opt_layout():
    return vgg_net_opt_layout(cfgs["B"], layouts["B"], "vgg_13")