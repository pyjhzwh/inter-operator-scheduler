from typing import Sequence
import itertools
import numpy as np
import ios
from ios.ir import *

def seqeq(seq1: Sequence, seq2: Sequence):
    """
    Compare whether two sequences are equal
    """
    for a, b in zip(seq1, seq2):
        if a != b:
            return False
    return True


def iter_subset(s: int, include_emtpy_set=False):
    """
    Iterate the subset of a set represented by the binary representation of s.
    """
    ss = s
    while ss != 0:
        yield ss
        ss = (ss - 1) & s
    if include_emtpy_set:
        yield 0


def get_conv_key(node: Node):
    assert(len(node.weight_shape) == 4)
    assert(len(node.stride) == 2)
    assert(len(node.padding) == 2)
    assert(node.weight_shape[1] == node.input_shape[0]) # in_c matches
    param = [node.input_shape[0], node.input_shape[1], node.input_shape[2],  # 0, 1, 2
        node.weight_shape[0], node.weight_shape[1], node.weight_shape[2], node.weight_shape[3], # 3, 4, 5, 6
        node.stride[0], node.stride[1], node.padding[0], node.padding[1], node.act] # 7, 8, 9, 10, 11
    return param

def create_conv_graph_given_layout(conv_param: list, input_layout: str, output_layout: str):
    v = ios.placeholder(output_shape=(conv_param[:3]), layout=input_layout)
    block = ios.Block(enter_node=v.node)
    ios.conv2d(block, inputs=[[v]], out_channels=conv_param[3],
        kernel=(conv_param[5:7]), stride=(conv_param[7:9]), padding=(conv_param[9:11]),
        act=conv_param[11], layout=output_layout, is_exit=True)
    
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph

def conv_latency(conv_param: list):
    layouts = ["NCHW", "NHWC"]
    # print(conv_param)
    latencies = []
    for input_layout, output_layout in itertools.product(layouts, layouts):
        graph = create_conv_graph_given_layout(conv_param, input_layout, output_layout)

        graph.sequential_schedule()
        seq_latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=10)
        latencies.append(np.mean(seq_latency))

    return latencies

def get_transform_conv_blacklist(graph: Graph):
    blacklist = []
    conv_nodes = []
    for block in graph.blocks:
        for node in block.inner_nodes + [block.exit_node]:
            if isinstance(node, Conv) and node.groups == 1:
                conv_nodes.append(node)

    conv_keys = []
    for node in conv_nodes:
        conv_key = get_conv_key(node)
        if conv_key not in conv_keys:
            latency = conv_latency(conv_key)
            conv_keys.append(conv_key)
            # NCHW-NCHW(default is the best layout)
            if np.argmin(latency) == 0:
                # no need to try different layout during optimization
                blacklist.append(conv_key)

    return blacklist
