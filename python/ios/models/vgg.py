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


def vgg_net(cfg, name, use_cuda=False, use_tc=False, layout="NCHW"):
    assert not (use_cuda and use_tc), "could not force using tc (tensor core) when force using cuda core "
    reset_name()

    pv = placeholder(output_shape=(3, 224, 224))
    block = Block(pv.node, None, [], None)

    v = pv
    for c in cfg:
        if c == 'M':
            v = pool2d(block, [[v]], pool_type='max', kernel=(2, 2), stride=(2, 2))
        else:
            v = conv2d(block, [[v]], c, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act="relu",
                disable_tc=use_cuda, use_tc=use_tc, layout=layout)
    v = pool2d(block, [[v]], pool_type='global_avg', is_exit=True)
    graph = Graph(name, pv.node, [block])
    graph.init_weights()
    return graph


def vgg_net_opt_layout(cfg, layout, name):
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

# use_cuda_tc: 0 - use cuda core; 1 - use tensor core; 2 - use both
def vgg_net_split_batch(cfg, name, batch_size, use_cuda_tc = 0):
    reset_name()

    pv = placeholder(output_shape=(3, 224, 224), layout="NHWC" if use_cuda_tc == 1 else "NCHW")
    block = Block(pv.node, None, [], None)

    out_split0 = split_batch(block, inputs=[[pv]], batch_begin=0, batch_end=batch_size//2)
    out_split1 = split_batch(block, inputs=[[pv]], batch_begin=batch_size//2, batch_end=batch_size)
    # v = pv
    for c in cfg:
        if c == 'M':
            out_split0 = pool2d(block, [[out_split0]], pool_type='max', kernel=(2, 2), stride=(2, 2))
            out_split1 = pool2d(block, [[out_split1]], pool_type='max', kernel=(2, 2), stride=(2, 2))
        else:
            out_split0 = conv2d(block, [[out_split0]], c, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act="relu",
                disable_tc= True if (use_cuda_tc == 0 or use_cuda_tc == 2) else False,
                use_tc = True if (use_cuda_tc == 1) else False,
                layout= "NCHW" if (use_cuda_tc == 0 or use_cuda_tc == 2) else "NHWC")
            out_split1 = conv2d(block, [[out_split1]], c, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act="relu",
                disable_tc= True if (use_cuda_tc == 0) else False,
                use_tc = True if (use_cuda_tc == 1 or use_cuda_tc == 2) else False,
                layout= "NCHW" if (use_cuda_tc == 0) else "NHWC")
    out_split0 = pool2d(block, [[out_split0]], pool_type='global_avg')
    out_split1 = pool2d(block, [[out_split1]], pool_type='global_avg')
    out_tensor = identity(block, inputs=[[out_split0], [out_split1]], is_exit=True)
    graph = Graph(name, pv.node, [block])
    graph.init_weights()
    return graph

def vgg_11(use_cuda=False, use_tc=False, layout="NCHW"):
    return vgg_net(cfgs['A'], 'vgg_11', use_cuda, use_tc, layout)


def vgg_13(use_cuda=False, use_tc=False, layout="NCHW"):
    return vgg_net(cfgs['B'], 'vgg_13', use_cuda, use_tc, layout)


def vgg_16(use_cuda=False, use_tc=False, layout="NCHW"):
    return vgg_net(cfgs['D'], 'vgg_16', use_cuda, use_tc, layout)


def vgg_19(use_cuda=False, use_tc=False, layout="NCHW"):
    return vgg_net(cfgs['E'], 'vgg_19', use_cuda, use_tc, layout)

def vgg_11_split_batch(batch_size, use_cuda_tc):
    return vgg_net_split_batch(cfgs['A'], 'vgg_11', batch_size=batch_size, use_cuda_tc=use_cuda_tc)

def vgg_13_split_batch(batch_size, use_cuda_tc):
    return vgg_net_split_batch(cfgs['B'], 'vgg_13', batch_size=batch_size, use_cuda_tc=use_cuda_tc)

def vgg_16_split_batch(batch_size, use_cuda_tc):
    return vgg_net_split_batch(cfgs['D'], 'vgg_16', batch_size=batch_size, use_cuda_tc=use_cuda_tc)

def vgg_19_split_batch(batch_size, use_cuda_tc):
    return vgg_net_split_batch(cfgs['E'], 'vgg_19', batch_size=batch_size, use_cuda_tc=use_cuda_tc)

def vgg_11_opt_layout():
    return vgg_net_opt_layout(cfgs["A"], layouts["A"], "vgg_11")

def vgg_13_opt_layout():
    return vgg_net_opt_layout(cfgs["B"], layouts["B"], "vgg_13")