from termios import VT1
import numpy as np
import ios
from ios import reset_name


def sample_network():
    reset_name()
    v = ios.placeholder(output_shape=(3, 180, 180), layout="NCHW")
    block = ios.Block(enter_node=v.node)
    v1 = ios.conv2d(block, inputs=[[v]], out_channels=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW")
    v2 = ios.conv2d(block, inputs=[[v1]], out_channels=128, kernel=(1, 1), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW", is_exit=True)
    # out = ios.identity(block, inputs=[[v1]], is_exit=True)
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph


def split1_network():
    reset_name()
    v = ios.placeholder(output_shape=(3, 180, 180), layout="NCHW")
    block = ios.Block(enter_node=v.node)
    v1_1 = ios.conv2d(block, inputs=[[v]], out_channels=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW")
    v1_2 = ios.conv2d(block, inputs=[[v]], out_channels=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW")
    v1 = ios.identity(block, inputs=[[v1_1], [v1_2]])
    v2_1 = ios.conv2d(block, inputs=[[v1]], out_channels=64, kernel=(1, 1), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW")
    v2_2 = ios.conv2d(block, inputs=[[v1]], out_channels=64, kernel=(1, 1), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW")
    # v1 = ios.conv2d(block, inputs=[[v1]], out_channels=750, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NHWC")
    out = ios.identity(block, inputs=[[v2_1], [v2_2]], is_exit=True)
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph


def split2_network():
    v = ios.placeholder(output_shape=(128, 112, 112), layout="NCHW")
    block = ios.Block(enter_node=v.node)
    v0 = ios.transform(block, inputs=[[v]], dst_layout="NHWC")
    v1 = ios.conv2d(block, inputs=[[v0]], out_channels=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NHWC", disable_tc=False)
    v2 = ios.conv2d(block, inputs=[[v]], out_channels=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), act='relu', layout="NCHW", disable_tc=True)
    out = ios.identity(block, inputs=[[v1], [v2]], is_exit=True)
    graph = ios.Graph(name="demo", input=v.node, blocks=[block])
    graph.init_weights()
    return graph

warmup=10
repeat=100

# define computation graph
graph = sample_network()
graph1 = split1_network()
# graph2 = split2_network()

# optimize execution schedule
optimized_graph0 = ios.optimize(graph, batch_size=1, opt_type='dp_parallel', compute_weight=True)
# optimized_graph1 = ios.optimize(graph1, batch_size=1, opt_type='dp_parallel', compute_weight=True)
stage_list1 = [[([[0], [1]], 'parallel'), ([[2]], 'parallel'), ([[3], [4]], 'parallel'), ([[5]], 'parallel')]]
optimized_graph1 = ios.graph_schedule_by_stage_list(graph1, stage_list1, compute_weight=True)
# optimized_graph2 = ios.optimize(graph2, batch_size=1, opt_type='dp_parallel', compute_weight=True)

# graph.sequential_schedule()
# graph1.sequential_schedule()
# graph2.sequential_schedule()

# # measure latency
opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph0, batch_size=1, warmup=warmup, repeat=repeat, profile_stage=True)
print(optimized_graph0)
print(f'Sequential schedule: {np.mean(opt_latency):.3f} ms')
print(f'      Stage latency: {np.mean(np.array(stage_latency).reshape(repeat, -1), axis=0)}\n')


opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph1, batch_size=1, warmup=warmup, repeat=repeat, profile_stage=True)
print(optimized_graph1)
print(f'Optimized schedule1: {np.mean(opt_latency):.3f} ms')
print(f'     Stage latency1: {np.mean(np.array(stage_latency).reshape(repeat, -1), axis=0)}')

# opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph2, batch_size=1, warmup=warmup, repeat=repeat, profile_stage=True)
# print(optimized_graph2)
# print(f'Optimized schedule2: {np.mean(opt_latency):.3f} ms')
# print(f'     Stage latency2: {np.mean(np.array(stage_latency).reshape(repeat, -1), axis=0)}')

# # inference on ios runtime
# dummy_inputs = np.random.randn(1, 375, 15, 15)
# output = ios.ios_runtime.graph_inference(optimized_graph, batch_size=1, input=dummy_inputs)

