import gzip
import re
from collections import Counter
from datetime import datetime

from Bio import SeqIO
from Bio.Seq import Seq
import dgl
import networkx as nx
import edlib
from tqdm import tqdm

import Gtm_utils.labels as labels  # type: ignore


def get_neighbors(graph):
    """Return neighbors/successors for each node in the graph."""
    neighbor_dict = {i.item(): [] for i in graph.nodes()}
    for src, dst in zip(graph.edges()[0], graph.edges()[1]):
        neighbor_dict[src.item()].append(dst.item())
    return neighbor_dict


def get_predecessors(graph):
    """Return predecessors for each node in the graph."""
    predecessor_dict = {i.item(): [] for i in graph.nodes()}
    for src, dst in zip(graph.edges()[0], graph.edges()[1]):
        predecessor_dict[dst.item()].append(src.item())
    return predecessor_dict


def get_edges(graph):
    """Return edge index for each edge in the graph."""
    edges_dict = {}
    for idx, (src, dst) in enumerate(zip(graph.edges()[0], graph.edges()[1])):
        src, dst = src.item(), dst.item()
        edges_dict[(src, dst)] = idx
    return edges_dict


def print_pairwise(graph, path):
    """Outputs the graph into a pairwise TXT format."""
    with open(path, 'w') as f:
        for src, dst in zip(graph.edges()[0], graph.edges()[1]):
            f.write(f'{src}\t{dst}\n')


def calculate_similarities(edge_ids, read_seqs, overlap_lengths, read_lengths):
    overlap_similarities = {}
    edit_distances = {}
    prefix_length_ratios = {}
    for src, dst in tqdm(edge_ids.keys(), ncols=120):
        ol_length = overlap_lengths[(src, dst)]
        if ol_length > 0:
            read_src = read_seqs[src]
            read_dst = read_seqs[dst]
            result = edlib.align(read_src[-ol_length:], read_dst[:ol_length])
            edit_distance = result['editDistance']
            overlap_similarities[(src, dst)] = 1 - edit_distance / ol_length if ol_length != 0 else 0.0
            edit_distances[(src, dst)] = edit_distance
            src_length = read_lengths[src]
            prefix_ratio = ol_length / src_length if src_length > 0 else 0.0
            prefix_length_ratios[(src, dst)] = prefix_ratio
        else:
            overlap_similarities[(src, dst)] = 0.5
            edit_distances[(src, dst)] = 0
            prefix_length_ratios[(src, dst)] = 0.0
    return overlap_similarities, edit_distances, prefix_length_ratios


def only_from_gfa(gfa_path, training=False, reads_path=None, get_similarities=False, paf_path=None):

    graph_nx = nx.DiGraph()
    read_to_node, node_to_read = {}, {}
    read_to_node2 = {}
    read_lengths, read_seqs = {}, {}  
    read_strands, read_starts, read_ends, read_chrs = {}, {}, {}, {}  
    edge_ids, prefix_lengths, overlap_lengths = {}, {}, {} 
    overlap_similarities, edit_distances, prefix_length_ratios = {}, {}, {} 
    no_seqs_flag = False
    read_headers = {}  

  
    if training:
        if reads_path is None:
            raise ValueError("训练模式下必须提供'reads_path'以获取序列头信息！")
       
        print(f'训练模式：从{reads_path}加载序列头信息...')
        if reads_path.endswith('gz'):
            if reads_path.endswith(('fasta.gz', 'fna.gz', 'fa.gz')):
                filetype = 'fasta'
            elif reads_path.endswith(('fastq.gz', 'fnq.gz', 'fq.gz')):
                filetype = 'fastq'
            with gzip.open(reads_path, 'rt') as handle:
                read_headers = {read.id: read.description for read in SeqIO.parse(handle, filetype)}
        else:
            if reads_path.endswith(('fasta', 'fna', 'fa')):
                filetype = 'fasta'
            elif reads_path.endswith(('fastq', 'fnq', 'fq')):
                filetype = 'fastq'
            read_headers = {read.id: read.description for read in SeqIO.parse(reads_path, filetype)}
        print(f'成功加载{len(read_headers)}条序列头信息')

    time_start = datetime.now()
    print(f'Starting to loop over GFA')
    with open(gfa_path) as f:
        node_idx = 0
        edge_idx = 0
        all_lines = f.readlines()
        line_idx = 0
        while line_idx < len(all_lines):
            line = all_lines[line_idx]
            line_idx += 1
            line = line.strip().split()
            if not line:
                continue  

            if line[0] == 'S':
              
                tag, id, sequence, length = line[:4]
                if sequence == '*':
                    no_seqs_flag = True
                sequence = Seq(sequence)  
                length = int(length[5:])  

                
                if len(sequence) > 0:
                    gc_count = sequence.count('G') + sequence.count('C')
                    gc_content = gc_count / len(sequence)
                else:
                    gc_content = 0.0

                
                real_idx = node_idx
                virt_idx = node_idx + 1
                graph_nx.add_node(real_idx)
                graph_nx.add_node(virt_idx)

                
                graph_nx.nodes[real_idx]['gc_content'] = gc_content
                graph_nx.nodes[virt_idx]['gc_content'] = gc_content
                graph_nx.nodes[real_idx]['read_length'] = length
                graph_nx.nodes[virt_idx]['read_length'] = length

                read_to_node[id] = (real_idx, virt_idx)
                node_to_read[real_idx] = id
                node_to_read[virt_idx] = id

               
                read_seqs[real_idx] = str(sequence)
                read_seqs[virt_idx] = str(sequence.reverse_complement())
                read_lengths[real_idx] = length
                read_lengths[virt_idx] = length

                
                if id.startswith('utg'):
                    ids = []
                    while line_idx < len(all_lines):
                        next_line = all_lines[line_idx].strip().split()
                        if next_line[0] != 'A':
                            break
                        line_idx += 1
                        utg_to_read = next_line[4]
                        read_orientation = next_line[3]
                        ids.append((utg_to_read, read_orientation))
                        read_to_node2[utg_to_read] = (real_idx, virt_idx)
                    id = ids
                    node_to_read[real_idx] = id
                    node_to_read[virt_idx] = id

                
                if training:
                    if isinstance(id, list):
                        strands = []
                        starts = []
                        ends = []
                        chromosomes = []
                        for id_r, id_o in id:
                            
                            if id_r not in read_headers:
                                raise KeyError(f"序列ID '{id_r}' 未在reads_path中找到！")
                            description = read_headers[id_r]
                            strand_fasta = re.findall(r'strand=(\+|\-)', description)[0]
                            strand_fasta = 1 if strand_fasta == '+' else -1
                            strand_gfa = 1 if id_o == '+' else -1
                            strands.append(strand_fasta * strand_gfa)
                            starts.append(int(re.findall(r'start=(\d+)', description)[0]))
                            ends.append(int(re.findall(r'end=(\d+)', description)[0]))
                            chromosome = re.findall(r'chr=([0-9XYM]+)', description)[0]
                            if chromosome == 'X':
                                chromosome = -1
                            elif chromosome == 'Y':
                                chromosome = -2
                            elif chromosome == 'M':
                                chromosome = -3
                            else:
                                chromosome = int(chromosome)
                            chromosomes.append(chromosome)
                     
                        strand = 1 if sum(strands) >= 0 else -1
                        start = min(starts)
                        end = max(ends)
                        chromosome = Counter(chromosomes).most_common()[0][0]
                    else:
                       
                        if id not in read_headers:
                            raise KeyError(f"序列ID '{id}' 未在reads_path中找到！")
                        description = read_headers[id]
                        strand = re.findall(r'strand=(\+|\-)', description)[0]
                        strand = 1 if strand == '+' else -1
                        start = int(re.findall(r'start=(\d+)', description)[0])
                        end = int(re.findall(r'end=(\d+)', description)[0])
                        chromosome = re.findall(r'chr=([0-9XYM]+)', description)[0]
                        if chromosome == 'X':
                            chromosome = -1
                        elif chromosome == 'Y':
                            chromosome = -2
                        elif chromosome == 'M':
                            chromosome = -3
                        else:
                            chromosome = int(chromosome)
                    
                    read_strands[real_idx], read_strands[virt_idx] = strand, -strand
                    read_starts[real_idx] = read_starts[virt_idx] = start
                    read_ends[real_idx] = read_ends[virt_idx] = end
                    read_chrs[real_idx] = read_chrs[virt_idx] = chromosome

                node_idx += 2  

            elif line[0] == 'L':
                
                if len(line) == 6:
                   
                    tag, id1, orient1, id2, orient2, cigar = line
                elif len(line) == 7:
                    
                    tag, id1, orient1, id2, orient2, cigar, _ = line
                    id1 = re.findall(r'(.*):\d-\d*', id1)[0]
                    id2 = re.findall(r'(.*):\d-\d*', id2)[0]
                elif len(line) == 8:
                    
                    tag, id1, orient1, id2, orient2, cigar, _, _ = line
                else:
                    raise Exception("Unknown GFA format!")

                
                try:
                    ol_length = int(cigar[:-1])  
                except ValueError:
                    print('Cannot convert CIGAR string into overlap length!')
                    raise ValueError

                if ol_length == 0:
                    continue  

            
                if orient1 == '+' and orient2 == '+':
                    src_real = read_to_node[id1][0]
                    dst_real = read_to_node[id2][0]
                    src_virt = read_to_node[id2][1]
                    dst_virt = read_to_node[id1][1]
                elif orient1 == '+' and orient2 == '-':
                    src_real = read_to_node[id1][0]
                    dst_real = read_to_node[id2][1]
                    src_virt = read_to_node[id2][0]
                    dst_virt = read_to_node[id1][1]
                elif orient1 == '-' and orient2 == '+':
                    src_real = read_to_node[id1][1]
                    dst_real = read_to_node[id2][0]
                    src_virt = read_to_node[id2][1]
                    dst_virt = read_to_node[id1][0]
                elif orient1 == '-' and orient2 == '-':
                    src_real = read_to_node[id1][1]
                    dst_real = read_to_node[id2][1]
                    src_virt = read_to_node[id2][0]
                    dst_virt = read_to_node[id1][0]
                else:
                    continue  

                
                graph_nx.add_edge(src_real, dst_real)
                graph_nx.add_edge(src_virt, dst_virt)

                
                edge_ids[(src_real, dst_real)] = edge_idx
                edge_ids[(src_virt, dst_virt)] = edge_idx + 1
                edge_idx += 2

                overlap_lengths[(src_real, dst_real)] = ol_length
                overlap_lengths[(src_virt, dst_virt)] = ol_length

                prefix_lengths[(src_real, dst_real)] = read_lengths[src_real] - ol_length
                prefix_lengths[(src_virt, dst_virt)] = read_lengths[src_virt] - ol_length

   
    if no_seqs_flag and reads_path is not None:
        print(f'从FASTA/Q文件加载序列...')
        if reads_path.endswith('gz'):
            if reads_path.endswith(('fasta.gz', 'fna.gz', 'fa.gz')):
                filetype = 'fasta'
            elif reads_path.endswith(('fastq.gz', 'fnq.gz', 'fq.gz')):
                filetype = 'fastq'
            with gzip.open(reads_path, 'rt') as handle:
                fastaq_seqs = {read.id: read.seq for read in SeqIO.parse(handle, filetype)}
        else:
            if reads_path.endswith(('fasta', 'fna', 'fa')):
                filetype = 'fasta'
            elif reads_path.endswith(('fastq', 'fnq', 'fq')):
                filetype = 'fastq'
            fastaq_seqs = {read.id: read.seq for read in SeqIO.parse(reads_path, filetype)}

        print(f'序列加载完成！')
        for node_id in tqdm(read_seqs.keys(), ncols=120):
            read_id = node_to_read[node_id]
            # 处理单元ig的ID列表
            if isinstance(read_id, list):
                read_id = read_id[0][0]
            seq = fastaq_seqs[read_id]
            read_seqs[node_id] = str(seq if node_id % 2 == 0 else seq.reverse_complement())
        print(f'DNA序列加载完成！')

   
    if get_similarities:
        print(f'计算相似度特征...')
        if not edge_ids:
            # 无边缘时返回空字典
            print("警告：GFA中未找到边，跳过相似度计算")
            overlap_similarities, edit_distances, prefix_length_ratios = {}, {}, {}
        else:
            # 计算并存储边特征
            overlap_similarities, edit_distances, prefix_length_ratios = calculate_similarities(
                edge_ids, read_seqs, overlap_lengths, read_lengths
            )
            for (src, dst) in edge_ids.keys():
                graph_nx.edges[(src, dst)]['overlap_similarity'] = overlap_similarities[(src, dst)]
                graph_nx.edges[(src, dst)]['edit_distance'] = edit_distances[(src, dst)]
                graph_nx.edges[(src, dst)]['prefix_length_ratio'] = prefix_length_ratios[(src, dst)]
        print(f'相似度计算完成！')

  
    nx.set_node_attributes(graph_nx, read_lengths, 'read_length')
    nx.set_node_attributes(graph_nx, {n: graph_nx.nodes[n]['gc_content'] for n in graph_nx.nodes}, 'gc_content')
    node_attrs = ['read_length', 'gc_content']

    nx.set_edge_attributes(graph_nx, prefix_lengths, 'prefix_length')
    nx.set_edge_attributes(graph_nx, overlap_lengths, 'overlap_length')
    edge_attrs = ['prefix_length', 'overlap_length']

    labels = None
    if training:
        nx.set_node_attributes(graph_nx, read_strands, 'read_strand')
        nx.set_node_attributes(graph_nx, read_starts, 'read_start')
        nx.set_node_attributes(graph_nx, read_ends, 'read_end')
        nx.set_node_attributes(graph_nx, read_chrs, 'read_chr')
        node_attrs.extend(['read_strand', 'read_start', 'read_end', 'read_chr'])

       
        unqique_chrs = set(read_chrs.values())
        if len(unqique_chrs) == 1:
            ms_pos, labels = labels.process_graph(graph_nx)
        else:
            ms_pos, labels = labels.process_graph_combo(graph_nx)
        nx.set_edge_attributes(graph_nx, labels, 'y')
        edge_attrs.append('y')

   
    if get_similarities and overlap_similarities:
        edge_attrs.extend(['overlap_similarity', 'edit_distance', 'prefix_length_ratio'])

   
    graph_dgl = dgl.from_networkx(graph_nx, node_attrs=node_attrs, edge_attrs=edge_attrs)

   
    predecessors = get_predecessors(graph_dgl)
    successors = get_neighbors(graph_dgl)
    edges = get_edges(graph_dgl)

    if len(read_to_node2) != 0:
        read_to_node = read_to_node2

  
    read_paf = False
    edge_paf_info = {}
    if read_paf and paf_path:
      
        pass  # 替换原代码中的"..."避免语法错误

    # 构建辅助信息字典
    auxiliary = {
        'pred': predecessors,
        'succ': successors,
        'reads': read_seqs,
        'edges': edges,
        'read_to_node': read_to_node,
    }
    if labels is not None:
        auxiliary['labels'] = labels
    if 'node_to_read' in locals():
        auxiliary['node_to_read'] = node_to_read
    if read_paf and edge_paf_info:
        auxiliary['edge_paf_info'] = edge_paf_info

    return graph_dgl, auxiliary