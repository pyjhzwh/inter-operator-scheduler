import numpy as np
import ios
from ios import reset_name
import argparse
from typing import List
import ios.models as ios_models
from ios.models import sample

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

def main(model_name: str, settings: List[str], batch_size:int):
    warmup=10
    repeat=100

    input_shape = (128, 64, 64)
    out_channel_list = [512, 512, 512]
    kernel_size_list = [3, 1, 1]

    ios_model = getattr(ios_models, model_name)
    for setting in settings:
        if "Split" not in setting:
            model = getattr(ios_model, f"sample_network_{setting}")
            # define computation graph
            graph = model(input_shape, out_channel_list, kernel_size_list)
            # optimize execution schedule
            optimized_graph = ios.optimize(graph, batch_size=batch_size, opt_type='dp_parallel', compute_weight=True)
            # measure latency
            opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph, batch_size=batch_size, warmup=warmup, repeat=repeat, profile_stage=True)
            # throughput = batch_size * graph.flops() / np.mean(opt_latency) / 1e9
            print(optimized_graph)
            # print(f'graph_NCHW throughput: {throughput:.1f} TFLOPS')
            print(f'graph_{setting} schedule: {np.mean(opt_latency):.3f} ms')
            print(f'      Stage latency: {np.mean(np.array(stage_latency).reshape(repeat, -1), axis=0)}\n')
        
        elif setting == "Split_non_TC" or setting == "Split_TC":
            model = getattr(ios_model, f"sample_network_{setting}")
            graph = model(input_shape, out_channel_list, kernel_size_list, batch_size)
            stage_list1 = gen_stage_list1(len(kernel_size_list))
            optimized_graph1 = ios.graph_schedule_by_stage_list(graph, stage_list1, compute_weight=True)
            opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph1, batch_size=batch_size, warmup=warmup, repeat=repeat, profile_stage=True)
            # throughput = batch_size * optimized_graph1.flops() / np.mean(opt_latency) / 1e9
            print(optimized_graph1)
            # print(f'Split-non-TC throughput: {throughput:.1f} TFLOPS')
            print(f'{setting} schedule: {np.mean(opt_latency):.3f} ms')
            print(f'        Stage latency: {np.mean(np.array(stage_latency).reshape(repeat, -1), axis=0)}')

        elif setting == "Split2":
            model = getattr(ios_model, f"sample_network_{setting}")
            graph2 = model(input_shape, out_channel_list, kernel_size_list, batch_size)
            stage_list2 = gen_stage_list2(len(kernel_size_list))
            optimized_graph2 = ios.graph_schedule_by_stage_list(graph2, stage_list2, compute_weight=True)
            opt_latency, stage_latency = ios.ios_runtime.graph_latency(optimized_graph2, batch_size=batch_size, warmup=warmup, repeat=repeat, profile_stage=True)
            print(optimized_graph2)
            # throughput = batch_size * graph2.flops() / np.mean(opt_latency) / 1e9
            # print(f'Split2 throughput: {throughput:.1f} TFLOPS')
            print(f'Split2 schedule: {np.mean(opt_latency):.3f} ms')
            print(f'  Stage latency: {np.mean(np.array(stage_latency).reshape(repeat, -1), axis=0)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'get execution time and throughput of Split batch'
    )
    parser.add_argument('-m', '--model', type=str, default="sample",
                        help="name of model")
    parser.add_argument('-s', '--setting', type=str, nargs='+', 
                        default=["NCHW", "NHWC", "non_TC", "TC",
                        "Split_non_TC", "Split_TC", "Split2"],
                        help="which setting to test")
    parser.add_argument('-b', '--batch', type=int, default="2",
                        help="batch size")
    args = parser.parse_args()
    main(args.model, args.setting, args.batch)