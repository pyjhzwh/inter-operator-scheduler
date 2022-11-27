import numpy as np
import ios
from ios import reset_name

# def sample_network():
#     v = ios.placeholder(output_shape=(512, 32, 32), layout="NCHW")
#     block = ios.Block(enter_node=v.node)
#     v_split0 = ios.split_batch(block, inputs=[[v]], batch_begin=0, batch_end=1)
#     v_split1 = ios.split_batch(block, inputs=[[v]], batch_begin=1, batch_end=5)
#     # v0 = ios.conv2d(block, inputs=[[v_split0]], out_channels=512, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NHWC")
#     # v1 = ios.conv2d(block, inputs=[[v_split1]], out_channels=512, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NHWC")
#     out = ios.identity(block, inputs=[[v_split0], [v_split1]], is_exit=True)  # concat v1, v2, and v3
#     graph = ios.Graph(name="demo", input=v.node, blocks=[block])
#     graph.init_weights()
#     return graph


# define computation graph
# graph = sample_network()

# # optimize execution schedule
# # optimized_graph = ios.optimize(graph, batch_size=5, opt_type='dp_parallel', compute_weight=True)

# # measure latency
# graph.sequential_schedule()
# seq_latency, stage_latency = ios.ios_runtime.graph_latency(graph, batch_size=5, repeat=6, profile_stage=True)
# print(graph)
# print(f'Sequential schedule: {np.mean(seq_latency):.3f} ms')
# print(f'      Stage latency: {np.mean(np.array(stage_latency).reshape(6, -1), axis=0)}\n')


def sample_network_NCHW(input_shape, out_channel_list, kernel_size_list):
    reset_name()
    v = ios.placeholder(output_shape=input_shape, layout="NCHW")
    block = ios.Block(enter_node=v.node)
    out_tensor = v
    for i in range(len(out_channel_list)):
        is_exit = (i == len(out_channel_list) - 1)
        out_tensor = ios.conv2d(
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
    # v1 = ios.conv2d(block, inputs=[[v]], out_channels=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW")
    # v2 = ios.conv2d(block, inputs=[[v1]], out_channels=128, kernel=(1, 1), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW", is_exit=True)
    # out = ios.identity(block, inputs=[[v1]], is_exit=True)
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph


def sample_network_NHWC(input_shape, out_channel_list, kernel_size_list):
    reset_name()
    v = ios.placeholder(output_shape=input_shape, layout="NHWC")
    block = ios.Block(enter_node=v.node)
    out_tensor = v
    for i in range(len(out_channel_list)):
        is_exit = (i == len(out_channel_list) - 1)
        out_tensor = ios.conv2d(
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
    # v1 = ios.conv2d(block, inputs=[[v]], out_channels=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW")
    # v2 = ios.conv2d(block, inputs=[[v1]], out_channels=128, kernel=(1, 1), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW", is_exit=True)
    # out = ios.identity(block, inputs=[[v1]], is_exit=True)
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph

def split_network1(input_shape, out_channel_list, kernel_size_list):
    """
    Split batch into 2 parts, do not specify whether run on tensor core or not
    """
    reset_name()
    v = ios.placeholder(output_shape=input_shape, layout="NCHW")
    block = ios.Block(enter_node=v.node)
    v_split0 = ios.split_batch(block, inputs=[[v]], batch_begin=0, batch_end=2)
    v_split1 = ios.split_batch(block, inputs=[[v]], batch_begin=2, batch_end=4)
    # v_split1 = ios.transform(block, inputs=[[v_split1]], dst_layout="NHWC")
    out_tensor_0 = v_split0
    out_tensor_1 = v_split1
    for i in range(len(out_channel_list)):
        is_exit = (i == len(out_channel_list) - 1)
        out_tensor_0 = ios.conv2d(
            block,
            inputs=[[out_tensor_0]],
            out_channels=out_channel_list[i],
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NCHW",
        )
        out_tensor_1 = ios.conv2d(
            block,
            inputs=[[out_tensor_1]],
            out_channels=out_channel_list[i],
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NCHW",
        )
        
    out_tensor = ios.identity(block, inputs=[[out_tensor_0], [out_tensor_1]], is_exit=is_exit)
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph


def split_network2(input_shape, out_channel_list, kernel_size_list):
    """
    Split batch into 2 parts, one run on tensor core, another part run on non-tensor core
    """
    reset_name()
    v = ios.placeholder(output_shape=input_shape, layout="NCHW")
    block = ios.Block(enter_node=v.node)
    v_non_tc = ios.split_batch(block, inputs=[[v]], batch_begin=0, batch_end=2)
    v_split1 = ios.split_batch(block, inputs=[[v]], batch_begin=2, batch_end=4)
    v_tc = ios.transform(block, inputs=[[v_split1]], dst_layout="NHWC")
    out_tensor_0 = v_non_tc
    out_tensor_1 = v_tc
    for i in range(len(out_channel_list)):
        is_exit = (i == len(out_channel_list) - 1)
        out_tensor_0 = ios.conv2d(
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
        out_tensor_1 = ios.conv2d(
            block,
            inputs=[[out_tensor_1]],
            out_channels=out_channel_list[i],
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NCHW",
            use_tc=True,
        )
        
    out_tensor = ios.identity(block, inputs=[[out_tensor_0], [out_tensor_1]], is_exit=is_exit)
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph

def gen_stage_list1(num_conv):
    stage_list = []
    for i in range(2):
        stage_list.append(([[i]], 'parallel'))
    for id in range(num_conv):
        stage_list.append(([[id*2+2],[id*2+3]], 'parallel'))
    stage_list.append(([[num_conv*2+2]], 'parallel'))

    return [stage_list]

def gen_stage_list2(num_conv):
    stage_list = []
    for i in range(3):
        stage_list.append(([[i]], 'parallel'))
    for id in range(num_conv):
        stage_list.append(([[id*2+3],[id*2+4]], 'parallel'))
    stage_list.append(([[num_conv*2+3]], 'parallel'))

    return [stage_list]

warmup=10
repeat=100
batch_size=4

input_shape = (128, 64, 64)
out_channel_list = [512, 512, 512]
kernel_size_list = [3, 1, 1]
# define computation graph
# graph_NCHW = sample_network_NCHW(input_shape, out_channel_list, kernel_size_list)
# graph_NHWC = sample_network_NHWC(input_shape, out_channel_list, kernel_size_list)
# graph1 = split_network1(input_shape, out_channel_list, kernel_size_list)
graph2 = split_network2(input_shape, out_channel_list, kernel_size_list)

# optimize execution schedule
# optimized_graph_NCHW = ios.optimize(graph_NCHW, batch_size=batch_size, opt_type='dp_parallel', compute_weight=True)
# optimized_graph_NHWC = ios.optimize(graph_NHWC, batch_size=batch_size, opt_type='dp_parallel', compute_weight=True)
# stage_list1 = gen_stage_list1(len(kernel_size_list))
# optimized_graph1 = ios.graph_schedule_by_stage_list(graph1, stage_list1, compute_weight=True)
stage_list2 = gen_stage_list2(len(kernel_size_list))
optimized_graph2 = ios.graph_schedule_by_stage_list(graph2, stage_list2, compute_weight=True)

# measure latency
# opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph_NCHW, batch_size=batch_size, warmup=warmup, repeat=repeat, profile_stage=True)
# print(optimized_graph_NCHW)
# print(f'optimized_graph_NCHW schedule: {np.mean(opt_latency):.3f} ms')
# print(f'      Stage latency: {np.mean(np.array(stage_latency).reshape(repeat, -1), axis=0)}\n')

# opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph_NHWC, batch_size=batch_size, warmup=warmup, repeat=repeat, profile_stage=True)
# print(optimized_graph_NHWC)
# print(f'optimized_graph_NHWC schedule: {np.mean(opt_latency):.3f} ms')
# print(f'      Stage latency: {np.mean(np.array(stage_latency).reshape(repeat, -1), axis=0)}\n')

# opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph1, batch_size=batch_size, warmup=warmup, repeat=repeat, profile_stage=True)
# print(optimized_graph1)
# print(f'optimized_graph1 schedule: {np.mean(opt_latency):.3f} ms')
# print(f'     Stage latency1: {np.mean(np.array(stage_latency).reshape(repeat, -1), axis=0)}')

opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph2, batch_size=batch_size, warmup=warmup, repeat=repeat, profile_stage=True)
print(optimized_graph2)
print(f'optimized_graph2 schedule: {np.mean(opt_latency):.3f} ms')
print(f'     Stage latency2: {np.mean(np.array(stage_latency).reshape(repeat, -1), axis=0)}')