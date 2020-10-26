from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tqdm import tqdm
import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt

class Topology(object):
    def __init__(self, config, data_dir='./data/'):
        self.topology_file = data_dir + config.topology_file
        self.shortest_paths_file = self.topology_file +'_shortest_paths'
        self.DG = nx.DiGraph()

        self.load_topology()
        self.calculate_paths()

    def load_topology(self):
        print('[*] Loading topology...', self.topology_file)
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':')+2:header.find('\t')])
        self.num_links = int(header[header.find(':', 10)+2:])
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c = link
            self.link_idx_to_sd[int(i)] = (int(s),int(d))
            self.link_sd_to_idx[(int(s),int(d))] = int(i)
            self.link_capacities[int(i)] = float(c)
            self.link_weights[int(i)] = int(w)
            self.DG.add_weighted_edges_from([(int(s),int(d),int(w))])
        
        assert len(self.DG.nodes()) == self.num_nodes and len(self.DG.edges()) == self.num_links

        f.close()
        #print('nodes: %d, links: %d\n'%(self.num_nodes, self.num_links))

        #nx.draw_networkx(self.DG)
        #plt.show()
    
    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>')+1:])
                self.pair_idx_to_sd.append((s,d))
                self.pair_sd_to_idx[(s,d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':')+1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(node_path).size
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx+3:]
        else:
            print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s,d))
                        self.pair_sd_to_idx[(s,d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(list(nx.all_shortest_paths(self.DG, s, d, weight='weight')))
                        line = str(s)+'->'+str(d)+': '+str(self.shortest_paths[-1])
                        f.writelines(line+'\n')
        
        assert self.num_pairs == self.num_nodes*(self.num_nodes-1)
        f.close()
        
        print('pairs: %d, nodes: %d, links: %d\n'\
                %(self.num_pairs, self.num_nodes, self.num_links))

  

class Traffic(object):
    def __init__(self, config, num_nodes, data_dir='./data/', is_training=False):
        if is_training:
            self.traffic_file = data_dir + config.topology_file + config.traffic_file
        else:
            self.traffic_file = data_dir + config.topology_file + config.test_traffic_file
        self.num_nodes = num_nodes
        self.load_traffic(config)

    def load_traffic(self, config):
        assert os.path.exists(self.traffic_file)
        print('[*] Loading traffic matrices...', self.traffic_file)
        f = open(self.traffic_file, 'r')
        traffic_matrices = []
        for line in f:
            volumes = line.strip().split(' ')
            total_volume_cnt = len(volumes)
            assert total_volume_cnt == self.num_nodes*self.num_nodes
            matrix = np.zeros((self.num_nodes, self.num_nodes))
            for v in range(total_volume_cnt):
                i = int(v/self.num_nodes)
                j = v%self.num_nodes
                if i != j:
                    matrix[i][j] = float(volumes[v])
            #print(matrix + '\n')
            traffic_matrices.append(matrix)

        f.close()
        self.traffic_matrices = np.array(traffic_matrices)

        tms_shape = self.traffic_matrices.shape
        self.tm_cnt = tms_shape[0]
        print('Traffic matrices dims: [%d, %d, %d]\n'%(tms_shape[0], tms_shape[1], tms_shape[2]))


    
class Environment(object):
    def __init__(self, config, is_training=False):
        self.data_dir = './data/'
        self.topology = Topology(config, self.data_dir)
        self.traffic = Traffic(config, self.topology.num_nodes, self.data_dir, is_training=is_training)
        self.traffic_matrices = self.traffic.traffic_matrices*100*8/300/1000    #kbps
        self.tm_cnt = self.traffic.tm_cnt
        self.traffic_file = self.traffic.traffic_file
        self.num_pairs = self.topology.num_pairs
        self.pair_idx_to_sd = self.topology.pair_idx_to_sd
        self.pair_sd_to_idx = self.topology.pair_sd_to_idx
        self.num_nodes = self.topology.num_nodes
        self.num_links = self.topology.num_links
        self.link_idx_to_sd = self.topology.link_idx_to_sd
        self.link_sd_to_idx = self.topology.link_sd_to_idx
        self.link_capacities = self.topology.link_capacities
        self.link_weights = self.topology.link_weights
        self.shortest_paths_node = self.topology.shortest_paths                         # paths consist of nodes
        self.shortest_paths_link = self.convert_to_edge_path(self.shortest_paths_node)  # paths consist of links

    def convert_to_edge_path(self, node_paths):
        edge_paths = []
        num_pairs = len(node_paths)
        for i in range(num_pairs):
            edge_paths.append([])
            num_paths = len(node_paths[i])
            for j in range(num_paths):
                edge_paths[i].append([])
                path_len = len(node_paths[i][j])
                for n in range(path_len-1):
                    e = self.link_sd_to_idx[(node_paths[i][j][n], node_paths[i][j][n+1])]
                    assert e>=0 and e<self.num_links
                    edge_paths[i][j].append(e)
                #print(i, j, edge_paths[i][j])

        return edge_paths
