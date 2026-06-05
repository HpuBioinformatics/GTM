import argparse
import os
import subprocess

from create_inference_graphs import create_inference_graph
from inference import inference


def step1_run_assembler(asm, reads, out, threads):
    print(f"Step 1: Running {asm} on {reads} to generate the graph")

    asm_out = f'{out}/{asm}/output'
    error_log = f'{asm_out}/raven_assembly_error.log'
    os.makedirs(asm_out, exist_ok=True)

    if asm == 'hifiasm':
        cmd = (
            # f'./vendor/hifiasm-0.18.8/hifiasm '
            # f'--prt-raw -o {asm_out}/asm -t{threads} -l0 {reads}'
            f'./vendor/hifiasm-0.18.8/hifiasm '
            f'--prt-raw -o {asm_out}/asm -t{threads} {reads}'
        )
        subprocess.run(cmd, shell=True, check=True)

    elif asm == 'raven':
        cmd1 = (
            f'./vendor/raven-1.8.1/build/bin/raven '
            f'-t {threads} {reads} --paf {asm_out}/alignments.paf '
            f'> {asm_out}/assembly.fasta 2> {error_log}'
        )
        subprocess.run(cmd1, shell=True, check=True)

        cmd2 = f'mv graph_1.gfa {asm_out}/graph_1.gfa'
        subprocess.run(cmd2, shell=True, check=True)

    else:
        raise ValueError(f"Unsupported assembler: {asm}. Choose 'hifiasm' or 'raven'.")


def step2_prepare_graph(asm, reads, out):
    print("Step 2: Preparing the graph for the inference")

    asm_out = f'{out}/{asm}/output'

    if asm == 'hifiasm':
        gfa = f'{asm_out}/asm.bp.p_utg.gfa'
        paf_path = None

    elif asm == 'raven':
        gfa = f'{asm_out}/graph_1.gfa'
        paf_path = f'{asm_out}/alignments.paf'

    else:
        raise ValueError(f"Unsupported assembler: {asm}")

    create_inference_graph(gfa, reads, out, asm, paf_path=paf_path)


def step3_run_inference(out, asm, model, device, exp_name=None):
    print(f"Step 3: Using the model {model} to run inference on {device}")

    inference(
        data_path=out,
        assembler=asm,
        model_path=model,
        savedir=os.path.join(out, asm),
        device=device,
        exp_name=exp_name,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-r',
        '--reads',
        required=True,
        type=str,
        help='Path to the reads'
    )

    parser.add_argument(
        '--asm',
        type=str,
        default='hifiasm',
        help='Assembler used [hifiasm|raven]'
    )

    parser.add_argument(
        '-o',
        '--out',
        type=str,
        default='.',
        help='Output directory'
    )

    parser.add_argument(
        '-t',
        '--threads',
        type=str,
        default='1',
        help='Number of threads to use'
    )

    parser.add_argument(
        '-m',
        '--model',
        type=str,
        default='weights/Gtm_weights.pt',
        help='Path to the model'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device for model inference, e.g. cpu or cuda:0'
    )

    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Experiment name for hyperparameter configuration'
    )

    args = parser.parse_args()

    reads = args.reads
    out = args.out
    threads = args.threads
    model = args.model
    asm = args.asm
    device = args.device
    exp_name = args.exp_name

    step1_run_assembler(asm, reads, out, threads)

    step2_prepare_graph(asm, reads, out)

    step3_run_inference(out, asm, model, device, exp_name)

    asm_dir = f'{out}/{asm}/assembly'

    print(f"\nDone!")
    print(f"Assembly saved in: {asm_dir}/0_assembly.fasta")
    print(f"Thank you for using Gtm!")
