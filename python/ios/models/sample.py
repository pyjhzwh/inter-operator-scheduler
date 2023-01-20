from .common import *

def sample_network_NCHW(input_shape, out_channel_list, kernel_size_list):
    reset_name()
    v = placeholder(output_shape=input_shape, layout="NCHW")
    block = Block(enter_node=v.node)
    out_tensor = v
    for i in range(len(out_channel_list)):
        is_exit = (i == len(out_channel_list) - 1)
        out_tensor = conv2d(
            block,
            inputs=[[out_tensor]],
            out_channels=out_channel_list[i],
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NCHW",
            is_exit=is_exit
        )
    graph = Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph


def sample_network_NHWC(input_shape, out_channel_list, kernel_size_list):
    reset_name()
    v = placeholder(output_shape=input_shape, layout="NHWC")
    block = Block(enter_node=v.node)
    out_tensor = v
    for i in range(len(out_channel_list)):
        is_exit = (i == len(out_channel_list) - 1)
        out_tensor = conv2d(
            block,
            inputs=[[out_tensor]],
            out_channels=out_channel_list[i],
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NHWC",
            is_exit=is_exit
        )
    graph = Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph

def sample_network_non_TC(input_shape, out_channel_list, kernel_size_list):
    reset_name()
    v = placeholder(output_shape=input_shape, layout="NCHW")
    block = Block(enter_node=v.node)
    out_tensor = v
    for i in range(len(out_channel_list)):
        is_exit = (i == len(out_channel_list) - 1)
        out_tensor = conv2d(
            block,
            inputs=[[out_tensor]],
            out_channels=out_channel_list[i],
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NCHW",
            is_exit=is_exit,
            disable_tc=True,
        )
    graph = Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph

def sample_network_TC(input_shape, out_channel_list, kernel_size_list):
    reset_name()
    v = placeholder(output_shape=input_shape, layout="NHWC")
    block = Block(enter_node=v.node)
    out_tensor = v
    for i in range(len(out_channel_list)):
        is_exit = (i == len(out_channel_list) - 1)
        out_tensor = conv2d(
            block,
            inputs=[[out_tensor]],
            out_channels=out_channel_list[i],
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NHWC",
            is_exit=is_exit,
            use_tc=True
        )
    graph = Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph

def sample_network_Split_non_TC(input_shape, out_channel_list, kernel_size_list, batch_size):
    """
    Split batch into 2 parts, each run on cuda core
    """
    reset_name()
    v = placeholder(output_shape=input_shape, layout="NCHW")
    block = Block(enter_node=v.node)
    v_split0 = split_batch(block, inputs=[[v]], batch_begin=0, batch_end=batch_size//2)
    v_split1 = split_batch(block, inputs=[[v]], batch_begin=batch_size//2, batch_end=batch_size)
    # v_split1 = transform(block, inputs=[[v_split1]], dst_layout="NHWC")
    out_tensor_0 = v_split0
    out_tensor_1 = v_split1
    for i in range(len(out_channel_list)):
        is_exit = (i == len(out_channel_list) - 1)
        out_tensor_0 = conv2d(
            block,
            inputs=[[out_tensor_0]],
            out_channels=out_channel_list[i],
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NCHW",
            disable_tc=True,
        )
        out_tensor_1 = conv2d(
            block,
            inputs=[[out_tensor_1]],
            out_channels=out_channel_list[i],
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NCHW",
            disable_tc=True,
        )
        
    out_tensor = identity(block, inputs=[[out_tensor_0], [out_tensor_1]], is_exit=is_exit)
    graph = Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph


def sample_network_Split_TC(input_shape, out_channel_list, kernel_size_list, batch_size):
    """
    Split batch into 2 parts, each run on tensor core
    """
    reset_name()
    v = placeholder(output_shape=input_shape, layout="NHWC")
    block = Block(enter_node=v.node)
    v_split0 = split_batch(block, inputs=[[v]], batch_begin=0, batch_end=batch_size//2)
    v_split1 = split_batch(block, inputs=[[v]], batch_begin=batch_size//2, batch_end=batch_size)
    out_tensor_0 = v_split0
    out_tensor_1 = v_split1
    for i in range(len(out_channel_list)):
        is_exit = (i == len(out_channel_list) - 1)
        out_tensor_0 = conv2d(
            block,
            inputs=[[out_tensor_0]],
            out_channels=out_channel_list[i],
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NHWC",
            use_tc=True,
        )
        out_tensor_1 = conv2d(
            block,
            inputs=[[out_tensor_1]],
            out_channels=out_channel_list[i],
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NHWC",
            use_tc=True,
        )
        
    out_tensor = identity(block, inputs=[[out_tensor_0], [out_tensor_1]], is_exit=is_exit)
    graph = Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph



def sample_network_Split2(input_shape, out_channel_list, kernel_size_list, batch_size):
    """
    Split batch into 2 parts, one run on tensor core, another part run on non-tensor core
    """
    reset_name()
    v = placeholder(output_shape=input_shape, layout="NCHW")
    block = Block(enter_node=v.node)
    v_non_tc = split_batch(block, inputs=[[v]], batch_begin=0, batch_end=batch_size//2)
    v_split1 = split_batch(block, inputs=[[v]], batch_begin=batch_size//2, batch_end=batch_size)
    v_tc = transform(block, inputs=[[v_split1]], dst_layout="NHWC")
    out_tensor_0 = v_non_tc
    out_tensor_1 = v_tc
    for i in range(len(out_channel_list)):
        is_exit = (i == len(out_channel_list) - 1)
        out_tensor_0 = conv2d(
            block,
            inputs=[[out_tensor_0]],
            out_channels=out_channel_list[i],
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NCHW",
            disable_tc=True,
        )
        out_tensor_1 = conv2d(
            block,
            inputs=[[out_tensor_1]],
            out_channels=out_channel_list[i],
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NHWC",
            use_tc=True,
        )
    out_tensor = identity(block, inputs=[[out_tensor_0], [out_tensor_1]], is_exit=is_exit)
    graph = Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph
