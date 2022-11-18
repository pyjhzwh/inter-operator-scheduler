from termios import VT1
import numpy as np
import ios
from ios import reset_name


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


def split1_network(input_shape, out_channel_list, kernel_size_list):
    reset_name()
    v = ios.placeholder(output_shape=input_shape, layout="NCHW")
    block = ios.Block(enter_node=v.node)
    out_tensor = v
    for i in range(len(out_channel_list)):
        is_exit = (i == len(out_channel_list) - 1)
        out_tensor_0 = ios.conv2d(
            block,
            inputs=[[out_tensor]],
            out_channels=out_channel_list[i]//2,
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NCHW",
        )
        out_tensor_1 = ios.conv2d(
            block,
            inputs=[[out_tensor]],
            out_channels=out_channel_list[i]//2,
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NCHW",
        )
        out_tensor = ios.identity(block, inputs=[[out_tensor_0], [out_tensor_1]], is_exit=is_exit)
    # v1_1 = ios.conv2d(block, inputs=[[v]], out_channels=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW")
    # v1_2 = ios.conv2d(block, inputs=[[v]], out_channels=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW")
    # v1 = ios.identity(block, inputs=[[v1_1], [v1_2]])
    # v2_1 = ios.conv2d(block, inputs=[[v1]], out_channels=64, kernel=(1, 1), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW")
    # v2_2 = ios.conv2d(block, inputs=[[v1]], out_channels=64, kernel=(1, 1), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW")
    # # v1 = ios.conv2d(block, inputs=[[v1]], out_channels=750, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NHWC")
    # out = ios.identity(block, inputs=[[v2_1], [v2_2]], is_exit=True)
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph


def split2_network(input_shape, out_channel_list, kernel_size_list):
    v = ios.placeholder(output_shape=input_shape, layout="NCHW")
    block = ios.Block(enter_node=v.node)
    out_tensor = v
    for i in range(len(out_channel_list)):
        is_exit = (i == len(out_channel_list) - 1)
        out_tensor_trans = ios.transform(block, inputs=[[out_tensor]], dst_layout="NHWC")
        out_tensor_0 = ios.conv2d(
            block,
            inputs=[[out_tensor]],
            out_channels=out_channel_list[i]//2,
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NCHW",
            disable_tc=True,
        )
        out_tensor_1 = ios.conv2d(
            block,
            inputs=[[out_tensor_trans]],
            out_channels=out_channel_list[i]//2,
            kernel=(kernel_size_list[i], kernel_size_list[i]),
            stride=(1, 1),
            padding=(1, 1),
            act='relu',
            layout="NHWC",
            disable_tc=False,
        )
        out_tensor = ios.identity(block, inputs=[[out_tensor_0], [out_tensor_1]], is_exit=is_exit)
    # v0 = ios.transform(block, inputs=[[v]], dst_layout="NHWC")
    # v1 = ios.conv2d(block, inputs=[[v0]], out_channels=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NHWC", disable_tc=False)
    # v2 = ios.conv2d(block, inputs=[[v]], out_channels=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW", disable_tc=True)
    # out = ios.identity(block, inputs=[[v1], [v2]], is_exit=True)
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph


def gen_stage_list(num_conv):
    stage_list = []
    for id in range(num_conv):
        stage_list.append(([[id*3],[id*3+1]], 'parallel'))
        stage_list.append(([[id*3+2]], 'parallel'))

    return [stage_list]


def gen_stage_list_with_transform(num_conv):
    stage_list = []
    for id in range(num_conv):
        stage_list.append(([[id*4]], 'parallel'))
        stage_list.append(([[id*4+1],[id*4+2]], 'parallel'))
        stage_list.append(([[id*4+3]], 'parallel'))

    return [stage_list]

warmup=10
repeat=100

input_shape = (128, 64, 64)
out_channel_list = [512, 512]
kernel_size_list = [3, 1]
# define computation graph
graph_NCHW = sample_network_NCHW(input_shape, out_channel_list, kernel_size_list)
graph_NHWC = sample_network_NHWC(input_shape, out_channel_list, kernel_size_list)
graph1 = split1_network(input_shape, out_channel_list, kernel_size_list)
graph2 = split2_network(input_shape, out_channel_list, kernel_size_list)

# optimize execution schedule
optimized_graph_NCHW = ios.optimize(graph_NCHW, batch_size=1, opt_type='dp_parallel', compute_weight=True)
optimized_graph_NHWC = ios.optimize(graph_NCHW, batch_size=1, opt_type='dp_parallel', compute_weight=True)
# optimized_graph1 = ios.optimize(graph1, batch_size=1, opt_type='dp_parallel', compute_weight=True)
# optimized_graph2 = ios.optimize(graph2, batch_size=1, opt_type='dp_parallel', compute_weight=True)
# stage_list1 = [[([[0], [1]], 'parallel'), ([[2]], 'parallel'), ([[3], [4]], 'parallel'), ([[5]], 'parallel')]]
stage_list1 = gen_stage_list(2)
optimized_graph1 = ios.graph_schedule_by_stage_list(graph1, stage_list1, compute_weight=True)
stage_list2 = gen_stage_list_with_transform(2)
optimized_graph2 = ios.graph_schedule_by_stage_list(graph2, stage_list2, compute_weight=True)


# graph.sequential_schedule()
# graph1.sequential_schedule()
# graph2.sequential_schedule()

# # measure latency
opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph_NHWC, batch_size=1, warmup=warmup, repeat=repeat, profile_stage=True)
print(optimized_graph_NHWC)
print(f'optimized_graph_NHWC schedule: {np.mean(opt_latency):.3f} ms')
print(f'      Stage latency: {np.mean(np.array(stage_latency).reshape(repeat, -1), axis=0)}\n')

opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph_NCHW, batch_size=1, warmup=warmup, repeat=repeat, profile_stage=True)
print(optimized_graph_NCHW)
print(f'optimized_graph_NCHW schedule: {np.mean(opt_latency):.3f} ms')
print(f'      Stage latency: {np.mean(np.array(stage_latency).reshape(repeat, -1), axis=0)}\n')


opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph1, batch_size=1, warmup=warmup, repeat=repeat, profile_stage=True)
print(optimized_graph1)
print(f'optimized_graph1 schedule: {np.mean(opt_latency):.3f} ms')
print(f'     Stage latency1: {np.mean(np.array(stage_latency).reshape(repeat, -1), axis=0)}')

opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph2, batch_size=1, warmup=warmup, repeat=repeat, profile_stage=True)
print(optimized_graph2)
print(f'optimized_graph2 schedule: {np.mean(opt_latency):.3f} ms')
print(f'     Stage latency2: {np.mean(np.array(stage_latency).reshape(repeat, -1), axis=0)}')

