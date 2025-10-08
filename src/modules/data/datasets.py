if __name__ == '__main__':
    from __init__ import external_dir, collation_dir, processed_dir, res_to_1hot
    # from retrievers import AlphaFold_Retriever
    from enm import ANM_Computer, TNM_Computer
    from encoders import ProtTrans_Encoder, ProteinBERT_Encoder
    # from persistence_image import PI_Computer
else:
    from .__init__ import external_dir, collation_dir, processed_dir, res_to_1hot
    # from .retrievers import AlphaFold_Retriever
    from .enm import ANM_Computer, TNM_Computer
    from .encoders import ProtTrans_Encoder, ProteinBERT_Encoder
    # from .persistence_image import PI_Computer

import os, torch, prody, warnings
import numpy as np
import torch_geometric as pyg

from tqdm import tqdm

df_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prody.confProDy(verbosity='none')

coupling_types = ['coord', 'codir', 'deform']
edge_types = ['contact'] + coupling_types

class Dataset(pyg.data.Dataset):

    def __init__(
        self,
        pdb_assembly_ids,
        annotations,
        version,
        sequence_embedding,
        enm_type,
        use_monomers,
        thresholds,
        merge_edge_types=False,
        time_limit=60,
        transform=None,
        device=df_device,
        entries_should_be_ready=False,
        # abort_on_process=False,
        rebuild=False,
        # float16_embeddings=True
    ):

        self.pdb_assemblies_to_process = pdb_assembly_ids
        self.annotations = annotations
        self.version = version
        self.sequence_embedding = sequence_embedding.lower()
        self.enm_type = enm_type.lower()
        self.use_monomers = use_monomers
        self.thresholds = thresholds
        self.merge_edge_types = merge_edge_types
        self.time_limit = time_limit
        self.entries_should_be_ready = entries_should_be_ready
        # self.abort_on_process = abort_on_process
        self.rebuild = rebuild
        # self.float16_embeddings = float16_embeddings
        self.threshold_name = (
            f'contact_{thresholds["contact"]}-'
            f'codir_{thresholds["codir"]}-'
            f'coord_{thresholds["coord"]}-'
            f'deform_{thresholds["deform"]}'
        )

        self.device = device

        ### COMMON DIRECTORIES
        self.v_process_dir = os.path.join(
            processed_dir,
            self.version
        )
        self.graph_dir = os.path.join(  # for graphs
            self.v_process_dir,
            'graphs' if not self.merge_edge_types else 'graphs_merged',
            'monomeric' if self.use_monomers else 'multimeric',
            self.threshold_name
        )
        self.embedding_dir = os.path.join(  # for sequence embeddings
            self.v_process_dir,
            'sequence_embeddings',
            self.sequence_embedding
        )

        os.makedirs(
            self.graph_dir,
            exist_ok=not self.entries_should_be_ready
        )
        os.makedirs(
            self.embedding_dir,
            exist_ok=not self.entries_should_be_ready
        )

        ### PROCESS ANNOTATIONS
        self.n_GO_terms = self.annotations.shape[1]
        assert len(self.pdb_assemblies_to_process) == len(self.annotations), (
            f'PDB assembly list have size {len(self.pdb_assemblies_to_process)} but '
            f'annotations have size {len(self.annotations)}'
        )

        ### INITIATE UTILITIES
        if enm_type == 'anm':
            self.enm_computer = ANM_Computer(
                cutoff=12,
                n_modes=20,
                use_monomers=self.use_monomers
            )
        elif enm_type == 'tnm':
            raise NotImplementedError('TNM not implemented yet')
            self.enm_computer = TNM_Computer()
        else:
            raise ValueError(f'Invalid ENM type: {enm_type}')

        ### CONSTRUCTOR OF PARENT CLASS
        super().__init__(self.raw_dir, transform, None, None)
        self.processed_assemblies = self.get_processable_assemblies()

        ### DEAL WITH LABELS
        self.update_annotations(self.processed_assemblies)

        ### STATISTICS AFTER PROCESSING
        print(' -> Number of unique accessions :', len(self.processed_assemblies))

        print(' -> Number of labels to predict :', self.n_GO_terms)

        print('Dataset instantiation complete.')

    @property
    def pos_weight(self):
        count = np.sum(self.annotations, axis=0)
        pos_weight = ( len(self.processed_file_names) - count ) / count
        return torch.from_numpy(pos_weight)

    @property
    def raw_dir(self):
        return os.path.join(external_dir, 'Protein_Data_Bank', 'pdb-biomt')

    @property
    def raw_file_names(self):
        return [f'{e}.pdb' for e in self.pdb_assemblies_to_process]

    @property
    def processed_dir(self):
        return self.graph_dir

    def get_processable_assemblies(self):

        if self.entries_should_be_ready:
            return np.array(self.pdb_assemblies_to_process, dtype=np.str_)

        not_processed = []
        for pdb_assembly_id in self.pdb_assemblies_to_process:
            graph_file = os.path.join(self.graph_dir, f'{pdb_assembly_id}.pt')
            embedding_file = os.path.join(self.embedding_dir, f'{pdb_assembly_id}.pt')
            if not (
                os.path.exists(graph_file) and os.path.exists(embedding_file)
            ):
                not_processed.append(pdb_assembly_id)

        failed_accessions = np.union1d(
            self.enm_computer.get_failed_entries(),
            not_processed
        )

        return np.setdiff1d(self.pdb_assemblies_to_process, failed_accessions)

    @property
    def processed_file_names(self):
        # for filename in np.char.add(self.pdb_assembly_ids, '.pt').tolist():
        #     if not os.path.exists(os.path.join(self.processed_dir, filename)):
        #         print(filename)
        return np.char.add(self.pdb_assemblies_to_process, '.pt').tolist()

    def download(self):
        raise NotImplementedError(
            'Please run preprocessing in the `data_curation` directory'
        )

    def process(self):

        if self.entries_should_be_ready:
            raise Exception(
                'Entering process(), aborting program since '
                '"entries_should_be_ready" is set to `True`.'
            )

        ################################################################
        # MODAL ANALYSIS
        ################################################################

        pdb_path_list = [
            os.path.join(self.raw_dir, pdb_file)
            for pdb_file in self.raw_file_names
        ]
        successful_accessions, _ = self.enm_computer.batch_run(
            self.pdb_assemblies_to_process,
            pdb_path_list,
        )

        print(
            ' -> Accessions with NMA results       :',
            len(successful_accessions),
        )

        ################################################################
        # BUILD GRAPHS
        ################################################################

        ### INSTANTIATE ENCODERS (TO BE DELETED AFTER USE)
        if self.sequence_embedding == 'proteinbert':
            encoder = ProteinBERT_Encoder()
        elif self.sequence_embedding == 'prottrans':
            encoder = ProtTrans_Encoder()
        else:
            raise ValueError(f'Invalid encoder {self.sequence_embedding}')

        pbar = tqdm(
            successful_accessions, dynamic_ncols=True, ascii=True
        )
        for pdb_assembly_id in pbar:
            # if pdb_assembly_id != '1A64-1':
            #     continue
            pbar.set_description(f'Graphs {pdb_assembly_id:<12s}')

            graph_file = os.path.join(self.graph_dir, f'{pdb_assembly_id}.pt')
            embedding_file = os.path.join(self.embedding_dir, f'{pdb_assembly_id}.pt')

            ### SKIP IF POSSIBLE
            if self.rebuild:
                if os.path.exists(graph_file):
                    os.remove(graph_file)
                if os.path.exists(embedding_file):
                    os.remove(embedding_file)
            elif (
                os.path.exists(graph_file) and
                os.path.exists(embedding_file)
            ):
                continue

            ### READ PDB FILE AND GET INFO
            # PDB files shoudl be preprocessed and cleaned
            path_to_pdb = os.path.join(self.raw_dir, f'{pdb_assembly_id}.pdb')
            atoms = prody.parsePDB(path_to_pdb, subset='ca')
            min_resnum = atoms.getResnums().min()

            ### FIND THE SEGMENT OF SEQUENCE SHARED BY ALL CHAINS
            all_sequences = []
            all_resnums = []
            for chain_idx, chain in enumerate(atoms.iterChains()):
                all_sequences.append(np.array([c for c in chain.getSequence(allres=False)]))
                all_resnums.append(chain.getResnums())

            # assert that resnums have one occurrence for all chains
            for chain_idx in range(atoms.numChains()):
                if len(np.unique(all_resnums[chain_idx])) != len(all_resnums[chain_idx]):
                    raise ValueError(
                        f'Chain {chain_idx} in {pdb_assembly_id} has duplicate resnums.'
                        f' This should be dealt with before calling this dataset.'
                    )

            # find intersection by resnums
            resnum_unique, resnum_count = np.unique([
                resnum for resnum_of_chain in all_resnums
                for resnum in resnum_of_chain
            ], return_counts=True)
            resnum_intersection = resnum_unique[resnum_count == atoms.numChains()]
            # print(len(resnum_intersection), 'residues in intersection')
            if len(resnum_intersection) == 0:
                tqdm.write(
                    f' -> {pdb_assembly_id} has no residues in the intersection, discarding.'
                    f' This should be dealt with before calling this dataset.'
                )
                continue

            # assert that all sequences are identical
            for chain1_idx in range(atoms.numChains()):
                for chain2_idx in range(chain1_idx+1, atoms.numChains()):

                    loc1 = np.isin(
                        all_resnums[chain1_idx], resnum_intersection, assume_unique=True
                    )
                    loc2 = np.isin(
                        all_resnums[chain2_idx], resnum_intersection, assume_unique=True
                    )
                    seq1_shared = all_sequences[chain1_idx][loc1]
                    seq2_shared = all_sequences[chain2_idx][loc2]

                    assert np.array_equal(seq1_shared, seq2_shared)

            if self.use_monomers:
                assert atoms.getChids()[0] == 'A'
                atoms = atoms.select('chain A')

            # proceed with residues in the intersection
            atoms.setResnums(atoms.getResnums() - min_resnum)
            atoms = atoms.select(
                f'resnum {resnum_intersection.min() - min_resnum} to '
                f'{resnum_intersection.max() - min_resnum}'
            )
            atoms.setResnums(atoms.getResnums() + min_resnum)
            resnames = atoms.select('chain A').getSequence(allres=False)
            sequence = ''.join(resnames)
            n_residues = len(sequence)

            ### CREATE HETERODATA OBJECT
            data = pyg.data.HeteroData()
            data.pdb_assembly_id = pdb_assembly_id
            data.num_nodes = n_residues

            try:
                resnum_intersection = resnum_intersection.astype(np.int32)
            except ValueError:
                tqdm.write(
                    f' -> {pdb_assembly_id} contains non-integer resids, discarding.'
                )
                continue

            # convert resIDs to one-hot-encoding
            resnames_1hot = np.zeros((n_residues, 20), dtype=np.int32)
            try:
                for j, resname in enumerate(resnames):
                    resnames_1hot[j, res_to_1hot[resname]] = 1
            except KeyError:
                tqdm.write(
                    f' -> {pdb_assembly_id} contains non-standard amino acids ({resname}), discarding.'
                )
                continue
            data['residue'].res1hot = torch.from_numpy(resnames_1hot)
            data['residue'].x = torch.from_numpy(resnames_1hot)

            # ### PERSISTENCE IMAGES
            # data.pi = torch.load(path_to_pi)

            ### BUILD CONTACT GRAPH
            # contact maps are built based on the position of Ca atoms
            all_edge_index, resnum_ij, distance = self.enm_computer.get_couplings(
                pdb_assembly_id,
                et_type='contact'
            )

            loc_to_include = (
                np.isin(resnum_ij[:,0], resnum_intersection, assume_unique=True) &
                np.isin(resnum_ij[:,1], resnum_intersection, assume_unique=True)
            )
            min_mat_idx = all_edge_index[loc_to_include,:].min()

            # dictionaries to convert between resnum and matrix index
            mat_idx, resnums, collapsed_idx = self.enm_computer.get_mapping(
                pdb_assembly_id
            )
            resnum2homo_idx = {
                resnums[idx]: collapsed_idx[idx] - min_mat_idx
                for idx in range(len(resnums))
            }
            multi2homo_idx = {
                mat_idx[idx]: collapsed_idx[idx] - min_mat_idx
                for idx in range(len(mat_idx))
            }
            collapse_fn = np.vectorize( lambda x: multi2homo_idx[x] )

            # remove unwanted residues
            all_edge_index = all_edge_index[loc_to_include,:]
            distance = distance[loc_to_include]

            edge_index = all_edge_index[distance <= 12].T # directed
            edge_index = np.unique(collapse_fn(edge_index), axis=1) # convert indices
            edge_index = np.hstack(( # undirected
                edge_index,
                np.flip(edge_index, axis=0)
            ))
            n_cont_edges = int(edge_index.shape[1] / 2)

            data['residue', 'contact', 'residue'].edge_index = torch.from_numpy(
                edge_index
            )
            threshold_values = {
                'contact': float(self.thresholds['contact'])
            }

            ### ADD BACKBONE CONNECTION
            edge_index = []
            for res_idx in range(len(resnum_intersection)-1):
                if resnum_intersection[res_idx+1] - resnum_intersection[res_idx] == 1:
                    edge_index.append((
                        resnum2homo_idx[int(resnum_intersection[res_idx])],
                        resnum2homo_idx[int(resnum_intersection[res_idx+1])]
                    ))
            edge_index = np.array(edge_index, dtype=np.int32).T # directed
            edge_index = np.unique(edge_index, axis=1)
            edge_index = np.hstack(( # undirected
                edge_index,
                np.flip(edge_index, axis=0)
            ))
            data['residue', 'backbone', 'residue'].edge_index = torch.from_numpy(
                edge_index
            )

            ### ADD DYNAMICAL COUPLING EDGES
            # 1. self-loops are not included
            # 2. indices are 0-based
            for ct in coupling_types:

                _, _, couplings = self.enm_computer.get_couplings(
                    pdb_assembly_id,
                    et_type=ct
                )
                couplings = couplings[loc_to_include]

                # dataset-wide thresholds
                if self.thresholds[ct].endswith('DCONT'):
                    raise NotImplementedError()
                    threshold = dataset_wide_thresholds[ct][self.thresholds[ct]]

                elif self.thresholds[ct].endswith('DN'):
                    raise NotImplementedError()
                    threshold = dataset_wide_thresholds[ct][self.thresholds[ct]]

                elif self.thresholds[ct].endswith('DSIGMA'):
                    raise NotImplementedError()
                    threshold = dataset_wide_thresholds[ct][self.thresholds[ct]]

                # entry-specific thresholds
                elif self.thresholds[ct].endswith('SIGMA'):
                    n_sigmas = float(self.thresholds[ct][:-5])
                    mean = np.mean(couplings)
                    sigma = np.std(couplings)
                    threshold = mean + sigma * n_sigmas

                elif self.thresholds[ct].endswith('N'):
                    n_N = float(self.thresholds[ct][:-1])
                    max_coupling = np.amax(couplings)
                    range_coupling = max_coupling - np.amin(couplings)
                    threshold = max_coupling - range_coupling * n_N / 100

                elif self.thresholds[ct].endswith('CONT'):
                    n_cont = float(self.thresholds[ct][:-4])
                    n_edges = int(n_cont * n_cont_edges)
                    threshold = np.sort(couplings)[-n_edges]

                elif self.thresholds[ct].endswith('PAIR'):
                    n_pair = float(self.thresholds[ct][:-4])
                    n_edges = int(
                        (n_residues * (n_residues - 1) / 2) * n_pair / 100
                    )
                    threshold = np.sort(couplings)[-n_edges]

                elif self.thresholds[ct].endswith('X'):
                    threshold_values[ct] = 'n/a'
                    continue

                # elif self.thresholds[ct] == 'DEF':
                #     raise NotImplementedError('DEF not implemented yet')
                #     threshold = 'tnm default'

                # elif self.thresholds[ct] == 'MEAN':
                #     threshold = np.mean(couplings)

                # elif self.thresholds[ct] == 'MEDIAN':
                #     threshold = np.median(couplings)

                # elif self.thresholds[ct].endswith('Q'):
                #     q = float(self.thresholds[ct][:-1])
                #     threshold = np.percentile(couplings, q/4*100)

                else:
                    try:
                        threshold = float(self.thresholds[ct])
                    except:
                        raise ValueError(
                            f'Invalid threshold for {ct}: {self.thresholds[ct]}'
                        )
                threshold_values[ct] = threshold

                discard = False
                if np.isnan(threshold):
                    tqdm.write(
                        f' -> {pdb_assembly_id} has nan as threshold for {ct}.'
                        f' This should be dealt with before calling this dataset.'
                    )
                    discard = True
                    raise
                    break

                edge_index = all_edge_index[couplings >= threshold].T # directed
                edge_index = np.unique(collapse_fn(edge_index), axis=1) # convert indices
                edge_index = np.hstack(( # undirected
                    edge_index,
                    np.flip(edge_index, axis=0)
                ))

                data['residue', ct, 'residue'].edge_index = torch.from_numpy(
                    edge_index
                )

            if discard:
                del resnum2homo_idx
                del multi2homo_idx
                del collapse_fn
                continue

            ### MERGE EDGES (IF SPECIFIED)
            if self.merge_edge_types:
                edge_types_to_merge = [
                    k for k, v in self.thresholds.items() if v != 'X'
                ]
                data['residue', 'merged', 'residue'].edge_index = torch.unique(
                    torch.cat(
                        [
                            data['residue', et, 'residue'].edge_index
                            for et in edge_types_to_merge
                        ],
                        dim=1,
                    ),
                    sorted=False,
                    dim=1,
                )

                diff_set = torch.tensor(list(
                    set(
                        (edge[0].item(), edge[1].item()) for edge in data['residue', 'backbone', 'residue'].edge_index.T
                    ) - set(
                        (edge[0].item(), edge[1].item()) for edge in data['residue', 'merged', 'residue'].edge_index.T
                    )
                ))

                data[
                    'residue', 'backbone-complementary', 'residue'
                ].edge_index = (
                    diff_set.T
                    if diff_set.numel()
                    else torch.empty(2,0, dtype=torch.int64)
                )

                # remove merged edge types
                for et in edge_types_to_merge:
                    del data['residue', et, 'residue']

            data.threshold_values = threshold_values

            assert data.is_undirected()

            torch.save(
                data, os.path.join(self.graph_dir, f'{pdb_assembly_id}.pt')
            )
            torch.save(
                encoder(sequence),
                os.path.join(self.embedding_dir, f'{pdb_assembly_id}.pt'),
            )

            del resnum2homo_idx
            del multi2homo_idx
            del collapse_fn

        # reclaim device memory
        del encoder

        processed_assemblies = self.get_processable_assemblies()
        print(
            ' -> Assemblies successfully processed :',
            len(processed_assemblies),
        )

    def update_annotations(self, processed_assemblies):

        # flag empty labels
        anno_count = self.annotations.sum(axis=0)
        empty_annotation_idx = np.where(anno_count == 0)[0]

        # only keep entries that are processed
        self.annotations = self.annotations[np.isin(
            self.pdb_assemblies_to_process, processed_assemblies
        )]

    def len(self):
        return len(self.processed_assemblies)

    def get(self, idx):

        filename = f'{self.processed_assemblies[idx]}.pt'

        # with warnings.catch_warnings():
        #     warnings.simplefilter(action='ignore', category=FutureWarning)
        data = torch.load(
            os.path.join(self.graph_dir, filename),
            weights_only=False
        )
        data['residue'].x = torch.load(
            os.path.join(self.embedding_dir, filename),
            weights_only=False
        )
        data.y = torch.tensor(
            self.annotations[idx], dtype=torch.bool
        ).reshape(1, -1)

        return data

if __name__ == '__main__':

    id_file = '../../data_curation/20250704-1 homo-multimer dataset (from scratch)/stats/OUT-6.entries_id.txt'
    anno_file = '../../data_curation/20250704-1 homo-multimer dataset (from scratch)/stats/OUT-6.labels_for_prediction.csv'

    pdb_assembly_ids = np.loadtxt(id_file, dtype=np.str_)
    print(pdb_assembly_ids.shape)

    annotations = np.loadtxt(
        anno_file,
        delimiter=',',
        dtype=np.int32
    )

    edge_policy = '1CONT'
    thresholds = {
        'contact': '12',
        'codir': edge_policy,
        'coord': edge_policy,
        'deform': edge_policy,
    }

    dataset_multimer = Dataset(
        pdb_assembly_ids,
        annotations=annotations,
        version='v0',
        sequence_embedding='ProtTrans',
        enm_type='anm',
        use_monomers=False,
        thresholds=thresholds,
        merge_edge_types=False,
        time_limit=60,
        transform=None,
        device=df_device,
        entries_should_be_ready=False,
        rebuild=False
    )
    print()

    dataset_monomer = Dataset(
        pdb_assembly_ids,
        annotations=annotations,
        version='v0',
        sequence_embedding='ProtTrans',
        enm_type='anm',
        use_monomers=True,
        thresholds=thresholds,
        merge_edge_types=False,
        time_limit=60,
        transform=None,
        device=df_device,
        entries_should_be_ready=False,
        rebuild=False
    )
    # dataloader = pyg.loader.DataLoader(
    #     dataset,
    #     batch_size=2,
    #     shuffle=False,
    #     num_workers=1,
    #     worker_init_fn=None,
    # )

    # for data_batch in dataloader:
    #     print(data_batch)
    #     break
