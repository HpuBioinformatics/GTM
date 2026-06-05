import re
import os
import pickle
import subprocess

import dgl
from dgl.data import DGLDataset

import graph_parser
from configs.config import get_config
from Gtm_utils.data_utils import preprocess_graph, add_positional_encoding, extract_hifiasm_contigs


class AssemblyGraphDataset(DGLDataset):
    def __init__(self, root, assembler, threads=32, generate=False, n_need=0):
      
        self.root = os.path.abspath(root)
        self.assembler = assembler
        self.threads = threads
        self.n_need = n_need
        self.assembly_dir = os.path.join(self.root, self.assembler)

 
        raw_dir = os.path.join(self.root, "raw")
        save_dir = os.path.join(self.assembly_dir, "processed")
        self.output_dir = os.path.join(self.assembly_dir, "output")
        self.info_dir = os.path.join(self.assembly_dir, "info")

        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(self.assembly_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(self.info_dir, exist_ok=True)

  
        config = get_config()
        raven_dir = config['raven_dir']
        self.raven_path = os.path.abspath(os.path.join(raven_dir, 'build/bin/raven'))
        hifiasm_dir = config['hifiasm_dir']
        self.hifiasm_path = os.path.abspath(os.path.join(hifiasm_dir, 'hifiasm'))

    
        super().__init__(name='assembly_graphs', raw_dir=raw_dir, save_dir=save_dir)

      
        self.graph_list = []

        if not generate:
            if not os.path.isdir(self.save_dir):
                print(f"[WARN] save_dir does not exist or is inaccessible: {self.save_dir}")
            else:
                for file in os.listdir(self.save_dir):
                   
                    if not file.endswith('.dgl'):
                        continue

                    m = re.match(r"(\d+)\.dgl$", file)
                    if not m:
                        print(f"[WARN] Ignoring file that cannot be parsed as an index: {file}")
                        continue
                    idx = int(m.group(1))

                    g_path = os.path.join(self.save_dir, file)
                    graphs, _ = dgl.load_graphs(g_path)
                    graph = graphs[0]

                    
                    graph = preprocess_graph(graph)
                    graph = add_positional_encoding(graph)

                    self.graph_list.append((idx, graph))

                self.graph_list.sort(key=lambda x: x[0])

            print(f'Number of graphs in the dataset: {len(self.graph_list)}')

    def has_cache(self):
        """Check if the raw data is already processed and stored."""
        if not os.path.isdir(self.save_dir):
            return False

        prc_files = {
            int(re.findall(r'(\d+).dgl', prc)[0])
            for prc in os.listdir(self.save_dir)
            if prc.endswith('.dgl') and re.findall(r'(\d+).dgl', prc)
        }
        needed_files = {i for i in range(self.n_need)}
        return len(needed_files - prc_files) == 0  # set difference

    def __len__(self):
       
        return len(self.graph_list)

    def __getitem__(self, idx):
        
        return self.graph_list[idx]

    def process(self):
       
        pass

class AssemblyGraphDataset_HiFi(AssemblyGraphDataset):

    def __init__(self, root, assembler='hifiasm', threads=1, generate=False, n_need=0):
        super().__init__(root=root, assembler=assembler, threads=threads, generate=generate, n_need=n_need)

    def process(self):
        """Process the raw data and save it on the disk."""
        assembler = 'hifiasm'
        print(f'hifiasm process')
        assert assembler in ('raven', 'hifiasm'), 'Choose either "raven" or "hifiasm" assembler'

        graphia_dir = os.path.join(self.assembly_dir, 'graphia')
        if not os.path.isdir(graphia_dir):
            os.mkdir(graphia_dir)

        prc_files = {int(re.findall(r'(\d+).dgl', prc)[0]) for prc in os.listdir(self.save_dir)}
        needed_files = {i for i in range(self.n_need)}
        diff = sorted(needed_files - prc_files)

        for cnt, idx in enumerate(diff):
            
            file_candidates = [
                f'{idx}.fasta.gz',
                f'{idx}.fasta',
                f'{idx}.fq.gz',
                f'{idx}.fastq.gz',
                f'{idx}.fastq'
            ]
           
            fastq = None
            for candidate in file_candidates:
                if candidate in os.listdir(self.raw_dir):
                    fastq = candidate
                    break
            if fastq is None:
                raise FileNotFoundError(f"No valid reads file found for index {idx} in {self.raw_dir}. Tried: {file_candidates}")
            print(f'\nStep {cnt}: generating graphs for reads in {fastq}')
            reads_path = os.path.abspath(os.path.join(self.raw_dir, fastq))
            print(f'Path to the reads: {reads_path}')
            print(f'Using assembler: {assembler}\n')
            
            # Raven
            if assembler == 'raven':
                subprocess.run(f'{self.raven_path} --disable-checkpoints --identity 0.99 -k29 -w9 -t{self.threads} -p0 {reads_path} > {idx}_assembly.fasta', shell=True, cwd=self.output_dir)
                subprocess.run(f'mv graph_1.gfa {idx}_raw_graph.gfa', shell=True, cwd=self.output_dir)
                gfa_path = os.path.join(self.output_dir, f'{idx}_raw_graph.gfa')

            # Hifiasm
          
            elif assembler == 'hifiasm':
                write_paf = False
                if not os.path.exists(reads_path):
                    raise FileNotFoundError(f"hifiasm input file does not exist: {reads_path}")
                
                if write_paf:
                    subprocess.run(
                        f'{self.hifiasm_path} --prt-raw --write-paf -o {idx}_asm -t{self.threads} -l0 {reads_path}',
                        shell=True,
                        cwd=self.output_dir,
                        check=True 
                    )
                    subprocess.run(f'mv {idx}_asm.ovlp.paf {idx}_ovlp.paf', shell=True, cwd=self.output_dir, check=True)
                    paf_path = os.path.join(self.output_dir, f'{idx}_ovlp.paf')
                else:
                    subprocess.run(
                        f'{self.hifiasm_path} --prt-raw -o {idx}_asm -t{self.threads} -l0 {reads_path}',
                        shell=True,
                        cwd=self.output_dir,
                        check=True  
                    )
                    paf_path = None
                
                
                expected_gfa = f'{idx}_asm.bp.raw.r_utg.gfa'
                if expected_gfa not in os.listdir(self.output_dir):
                    
                    possible_gfas = [f for f in os.listdir(self.output_dir) if f.startswith(f'{idx}_asm') and f.endswith('.gfa')]
                    if not possible_gfas:
                        raise FileNotFoundError(f"hifiasm failed to generate GFA files in output directory: {self.output_dir}")
                    expected_gfa = possible_gfas[0]  
                
                subprocess.run(f'mv {expected_gfa} {idx}_raw_graph.gfa', shell=True, cwd=self.output_dir, check=True)
                gfa_path = os.path.join(self.output_dir, f'{idx}_raw_graph.gfa')
                
                
                contig_gfa = f'{idx}_asm.bp.p_ctg.gfa'
                if contig_gfa not in os.listdir(self.output_dir):
                    possible_contig_gfas = [f for f in os.listdir(self.output_dir) if f.startswith(f'{idx}_asm') and 'p_ctg.gfa' in f]
                    if not possible_contig_gfas:
                        raise FileNotFoundError(f"Contig GFA file not found in output directory: {self.output_dir}")
                    contig_gfa = possible_contig_gfas[0]
              
                extract_hifiasm_contigs(self.output_dir, idx)
                subprocess.run(f'rm {self.output_dir}/{idx}_asm*', shell=True, check=True)

            print(f'\nAssembler generated the graph! Processing...')
            processed_path = os.path.join(self.save_dir, f'{idx}.dgl')
            graph, auxiliary = graph_parser.only_from_gfa(gfa_path, reads_path=reads_path, training=True, get_similarities=True, paf_path=paf_path)
            print(f'Parsed assembler output! Saving files...')

            dgl.save_graphs(processed_path, graph)
            for name, data in auxiliary.items():
                pickle.dump(data, open(f'{self.info_dir}/{idx}_{name}.pkl', 'wb'))

            graphia_path = os.path.join(graphia_dir, f'{idx}_graph.txt')
            graph_parser.print_pairwise(graph, graphia_path)
            print(f'Processing of graph {idx} generated from {fastq} done!\n')


class AssemblyGraphDataset_ONT(AssemblyGraphDataset):

    def __init__(self, root, assembler='raven', threads=1, generate=False, n_need=0):
        super().__init__(root=root, assembler=assembler, threads=threads, generate=generate, n_need=n_need)

    def process(self):
        """Process the raw data and save it on the disk."""
        assembler = 'raven'

        graphia_dir = os.path.join(self.assembly_dir, 'graphia')
        if not os.path.isdir(graphia_dir):
            os.mkdir(graphia_dir)

        # raw_files = {int(re.findall(r'(\d+).fast*', raw)[0]) for raw in os.listdir(self.raw_dir)}
        prc_files = {int(re.findall(r'(\d+).dgl', prc)[0]) for prc in os.listdir(self.save_dir)}
        needed_files = {i for i in range(self.n_need)}
        diff = sorted(needed_files - prc_files)

        for cnt, idx in enumerate(diff):
           
            file_candidates = [
                f'{idx}.fasta.gz',
                f'{idx}.fasta',
                f'{idx}.fq.gz',
                f'{idx}.fastq.gz',
                f'{idx}.fastq'
            ]
          
            fastq = None
            for candidate in file_candidates:
                if candidate in os.listdir(self.raw_dir):
                    fastq = candidate
                    break
            if fastq is None:
                raise FileNotFoundError(f"No valid reads file found for index {idx} in {self.raw_dir}. Tried: {file_candidates}")
            
            print(f'\nStep {cnt}: generating graphs for reads in {fastq}')
            reads_path = os.path.abspath(os.path.join(self.raw_dir, fastq))
            print(f'Path to the reads: {reads_path}')
            print(f'Using assembler: {assembler}')
            print(f'Other assemblers currently unavailable\n')
            
            # Raven
            if assembler == 'raven':
            
                subprocess.run(
                    f'{self.raven_path} --disable-checkpoints -k29 -w9 -t{self.threads} -p0 {reads_path} > {idx}_assembly.fasta',
                    shell=True,
                    cwd=self.output_dir
                )
              
                subprocess.run(f'mv graph_1.csv {idx}_graph_1.csv', shell=True, cwd=self.output_dir)
                subprocess.run(f'mv graph_1.gfa {idx}_raw_graph.gfa', shell=True, cwd=self.output_dir)
             
                gfa_path = os.path.join(self.output_dir, f'{idx}_raw_graph.gfa')
            print(f'\nAssembler generated the graph! Processing...')
            processed_path = os.path.join(self.save_dir, f'{idx}.dgl')
            graph, auxiliary = graph_parser.only_from_gfa(gfa_path, reads_path=reads_path, training=True, get_similarities=True)
            print(f'Parsed assembler output! Saving files...')

            dgl.save_graphs(processed_path, graph)
            for name, data in auxiliary.items():
                pickle.dump(data, open(f'{self.info_dir}/{idx}_{name}.pkl', 'wb'))

            graphia_path = os.path.join(graphia_dir, f'{idx}_graph.txt')
            graph_parser.print_pairwise(graph, graphia_path)
            print(f'Processing of graph {idx} generated from {fastq} done!\n')
