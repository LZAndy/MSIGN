import torch.nn as nn
from dgl.nn.pytorch import edge_softmax
import dgl
from dgl.utils import expand_as_pair
import dgl.function as fn

from HGC import HeteroGraphConv

class CovMeAggLayer(nn.Module):

    def __init__(self, input_dim, output_dim, drop=0.1):
        super(CovMeAggLayer, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(drop),
            nn.LeakyReLU(),
            nn.BatchNorm1d(output_dim)
        )

    def message(self, edges):
        return {'m': F.relu(edges.src['hn'] + edges.data['he'])}

    def forward(self, graph, node_feat, edge_feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(node_feat, graph)
            graph.srcdata['hn'] = feat_src
            graph.edata['he'] = edge_feat
            graph.update_all(self.message, fn.sum('m', 'neigh'))
            rst = feat_dst + graph.dstdata['neigh']
            rst = self.mlp(rst)

            return rst


class NonCovMeAggLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 bias=True):
        super(NonCovMeAggLayer, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
            graph.update_all(msg_fn, fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)

            rst = self.fc_self(h_self) + h_neigh

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            return rst

class MSIGN(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size, layer_num=3):
        super(MSIGN, self).__init__()

        self.convs = nn.ModuleList()

        for _ in range(layer_num):
            convl = CovMeAggLayer(hidden_feat_size, hidden_feat_size)
            convp = CovMeAggLayer(hidden_feat_size, hidden_feat_size)
            convlp = NonCovMeAggLayer(hidden_feat_size, hidden_feat_size, feat_drop=0.1)
            convpl = NonCovMeAggLayer(hidden_feat_size, hidden_feat_size, feat_drop=0.1)
            conv = HeteroGraphConv(
                    {
                        'intra_l' : convl,
                        'intra_p' : convp,
                        'inter_l2p' : convlp,
                        'inter_p2l': convpl
                    }
                )
            self.convs.append(conv)

        self.lin_node_l = nn.Linear(node_feat_size, hidden_feat_size)
        self.lin_node_p = nn.Linear(node_feat_size, hidden_feat_size)
        self.lin_edge_ll = nn.Linear(edge_feat_size, hidden_feat_size)
        self.lin_edge_pp = nn.Linear(edge_feat_size, hidden_feat_size)

        self.lin_edge_lp = nn.Linear(11, hidden_feat_size)
        self.lin_edge_pl = nn.Linear(11, hidden_feat_size)

        # atom-atom features
        self.phyfea_atompairs = AtomAtomMeAggLayer(hidden_feat_size, hidden_feat_size, hidden_feat_size)

        # chemical features aggregation
        self.chemfea_ligandpocket = LigandPocketChemGAggLayer(hidden_feat_size, hidden_feat_size, hidden_feat_size)
        self.chemfea_pocketligand = PocketLigandChemGAggLayer(hidden_feat_size, hidden_feat_size, hidden_feat_size)

    def forward(self, bg):
        atom_feats = bg.ndata['h']
        bond_feats = bg.edata['e']

        atom_feats = {
            'ligand':self.lin_node_l(atom_feats['ligand']),
            'pocket':self.lin_node_p(atom_feats['pocket'])
        }
        bond_feats = {
            ('ligand', 'intra_l', 'ligand'):self.lin_edge_ll(bond_feats[('ligand', 'intra_l', 'ligand')]),
            ('pocket', 'intra_p', 'pocket'):self.lin_edge_pp(bond_feats[('pocket', 'intra_p', 'pocket')]),       
            ('ligand', 'inter_l2p', 'pocket'):self.lin_edge_lp(bond_feats[('ligand', 'inter_l2p', 'pocket')]),    
            ('pocket', 'inter_p2l', 'ligand'):self.lin_edge_pl(bond_feats[('pocket', 'inter_p2l', 'ligand')]),        
        }

        bg.edata['e'] = bond_feats

        rsts = atom_feats
        for conv in self.convs:
            rsts = conv(bg, rsts)

        bg.nodes['ligand'].data['h'] = rsts['ligand']
        bg.nodes['pocket'].data['h'] = rsts['pocket']

        # atom-atom affinities
        atompairs_lp, atompairs_pl = self.phyfea_atompairs(bg)

        # chemical feature aggregation
        fea_lp = self.chemfea_ligandpocket(bg)
        fea_pl = self.chemfea_pocketligand(bg)

        return (atompairs_lp - fea_lp).view(-1), (atompairs_pl - fea_pl).view(-1)

class AtomAtomMeAggLayer(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size):
        super(AtomAtomMeAggLayer, self).__init__()
        self.prj_lp_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_lp_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_lp_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.prj_pl_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_pl_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_pl_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.fc_lp = nn.Linear(hidden_feat_size, 1)
        self.fc_pl = nn.Linear(hidden_feat_size, 1)

    def apply_interactions(self, edges):
        return {'i' : edges.data['e'] * edges.src['h'] * edges.dst['h']}

    def forward(self, g):
        with g.local_scope():
            node_ligand_feats = g.nodes['ligand'].data['h']
            node_pocket_feats = g.nodes['pocket'].data['h']
            edge_lp_feat = g.edges['inter_l2p'].data['e']
            edge_pl_feat = g.edges['inter_p2l'].data['e']

            g.nodes['ligand'].data['h'] = self.prj_lp_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.prj_lp_dst(node_pocket_feats)
            g.edges['inter_l2p'].data['e'] = self.prj_lp_edge(edge_lp_feat)
            g.apply_edges(self.apply_interactions, etype='inter_l2p')
            logit_lp = self.fc_lp(g.edges['inter_l2p'].data['i'])
            g.edges['inter_l2p'].data['logit_lp'] = logit_lp
            logit_lp = dgl.sum_edges(g, 'logit_lp', etype='inter_l2p')

            g.nodes['ligand'].data['h'] = self.prj_pl_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.prj_pl_dst(node_pocket_feats)
            g.edges['inter_p2l'].data['e'] = self.prj_pl_edge(edge_pl_feat)
            g.apply_edges(self.apply_interactions, etype='inter_p2l')
            logit_pl = self.fc_pl(g.edges['inter_p2l'].data['i'])
            g.edges['inter_p2l'].data['logit_pl'] = logit_pl
            logit_pl = dgl.sum_edges(g, 'logit_pl', etype='inter_p2l')

            return logit_lp, logit_pl

class LigandPocketChemGAggLayer(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size):
        super(LigandPocketChemGAggLayer, self).__init__()
        self.prj_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.w_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.w_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.w_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.lin_att = nn.Sequential(
            nn.PReLU(),
            nn.Linear(hidden_feat_size, 1)
        )

        self.fc = FC(hidden_feat_size, 200, 2, 0.1, 1)

    def get_weight(self, edges):
        w = edges.src['h'] + edges.dst['h'] + edges.data['e']
        w = self.lin_att(w)

        return {'w' : w}
    
    def apply_scores(self, edges):
        return {'l' : edges.data['a'] * edges.data['e'] * edges.src['h'] * edges.dst['h']}

    def forward(self, g):
        with g.local_scope():
            node_ligand_feats = g.nodes['ligand'].data['h']
            node_pocket_feats = g.nodes['pocket'].data['h']
            edge_feat = g.edges['inter_l2p'].data['e']

            g.nodes['ligand'].data['h'] = self.prj_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.prj_dst(node_pocket_feats)
            g.edges['inter_l2p'].data['e'] = self.prj_edge(edge_feat)
            g.apply_edges(self.get_weight, etype='inter_l2p')
            scores = edge_softmax(g['inter_l2p'], g.edges['inter_l2p'].data['w'])

            g.edges['inter_l2p'].data['a'] = scores
            g.nodes['ligand'].data['h'] = self.w_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.w_dst(node_pocket_feats)
            g.edges['inter_l2p'].data['e'] = self.w_edge(edge_feat)
            g.apply_edges(self.apply_scores, etype='inter_l2p')

            bias = self.fc(dgl.sum_edges(g, 'l', etype='inter_l2p')) 
            
            return bias

class PocketLigandChemGAggLayer(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size):
        super(PocketLigandChemGAggLayer, self).__init__()
        self.prj_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.prj_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.w_src = nn.Linear(node_feat_size, hidden_feat_size)
        self.w_dst = nn.Linear(node_feat_size, hidden_feat_size)
        self.w_edge = nn.Linear(edge_feat_size, hidden_feat_size)

        self.lin_att = nn.Sequential(
            nn.PReLU(),
            nn.Linear(hidden_feat_size, 1)
        )

        self.fc = FC(hidden_feat_size, 200, 2, 0.1, 1)

    def get_weight(self, edges):
        w = edges.src['h'] + edges.dst['h'] + edges.data['e']
        w = self.lin_att(w)

        return {'w' : w}
    
    def apply_scores(self, edges):
        return {'l' : edges.data['a'] * edges.data['e'] * edges.src['h'] * edges.dst['h']}

    def forward(self, g):
        with g.local_scope():
            node_ligand_feats = g.nodes['ligand'].data['h']
            node_pocket_feats = g.nodes['pocket'].data['h']
            edge_feat = g.edges['inter_p2l'].data['e']

            g.nodes['ligand'].data['h'] = self.prj_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.prj_dst(node_pocket_feats)
            g.edges['inter_p2l'].data['e'] = self.prj_edge(edge_feat)
            g.apply_edges(self.get_weight, etype='inter_p2l')
            scores = edge_softmax(g['inter_p2l'], g.edges['inter_p2l'].data['w'])

            g.edges['inter_p2l'].data['a'] = scores
            g.nodes['ligand'].data['h'] = self.w_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.w_dst(node_pocket_feats)
            g.edges['inter_p2l'].data['e'] = self.w_edge(edge_feat)
            g.apply_edges(self.apply_scores, etype='inter_p2l')

            bias = self.fc(dgl.sum_edges(g, 'l', etype='inter_p2l')) 
            
            return bias

class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h
