import os
import numpy as np
import ios
import parser
import argparse


def main(model_name: str, opt_type: str="dp_merge_parallel_transform"):
    graph = getattr(ios.models, model_name)()

    graph.sequential_schedule()
    latency = ios.ios_runtime.graph_latency(graph, batch_size=1, repeat=6)

    optimized_graph = ios.optimize(graph, batch_size=1, opt_type=opt_type)
    optimized_latency = ios.ios_runtime.graph_latency(optimized_graph, batch_size=1, repeat=6)

    print(optimized_graph)
    print(f" Sequential schedule: {np.mean(latency):.3f} ms")
    print(f"  Optimized schedule: {np.mean(optimized_latency):.3f} ms")
    os.makedirs("./outputs/", exist_ok=True)
    ios.draw(
        optimized_graph,
        fname=f"./outputs/{opt_type}_{graph.name}.png",
        label=f"Optimized Graph, Latency = {np.mean(optimized_latency):.3f}"
    )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("collect conv latency with differnt layout from model")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="name of model")
    parser.add_argument("-t", "--opt_type", type=str, default="dp_merge_parallel_transform",
                        help="opt_type: dp_merge_parallel_transform or dp_merge_parallel")
    args = parser.parse_args()
    main(args.model)
