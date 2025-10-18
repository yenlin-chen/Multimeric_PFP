if __name__ in ['__main__', 'enm']:
    from __init__ import collation_dir
else:
    from .__init__ import collation_dir

import os, shutil, subprocess, platform, json, prody
import numpy as np
from tqdm import tqdm

class ANM_Computer:

    '''Computes ANM features for a given protein structure.

    Calls ProDy internally to do NMA.
    '''

    def __init__(self, cutoff, n_modes, use_monomers):

        self.cutoff = cutoff
        self.n_modes = n_modes
        self.use_monomers = use_monomers

        self.anm_dir = os.path.join(
            collation_dir,
            'ANM',
            'monomeric' if self.use_monomers else 'multimeric',
        )
        os.makedirs(self.anm_dir, exist_ok=True)

        failure_filename = 'anm-failed_entries.tsv'
        self.failure_path = os.path.join(self.anm_dir, failure_filename)

        self.cached_couplings_entry = None
        self.cached_mapping_entry = None

    def get_failed_entries(self, return_reason=False):
        '''Returns entries for which ANM computation failed.'''

        if (os.path.exists(self.failure_path) and
            os.stat(self.failure_path).st_size!=0):
            entries = np.loadtxt(
                self.failure_path,
                dtype=np.str_,
                delimiter='\t'
            ).reshape((-1,2))
        else:
            entries = np.empty((0,2), dtype=np.str_)

        if return_reason:
            return np.unique(entries, axis=0)
        else:
            return np.unique(entries[:,0])

    def path_to_work(self, entry):
        '''Returns path to working directory for a given entry.'''
        return os.path.join(self.anm_dir, f'cutoff_{self.cutoff}', entry)

    def path_to_outputs(self, entry):

        work_dir = self.path_to_work(entry)
        filenames = {
            'couplings': 'coupling.csv',
            'mapping': 'mapping.csv',
        }
        return {k: os.path.join(work_dir, filenames[k]) for k in filenames}

    def run(self, entry_id, pdb_path):

        '''Runs TNM program on a given protein structure.

        Parameters
        ----------
        entry_id : str
            Entry identifier.
        pdb_path : str
            Path to PDB file.
        timeout : int, optional
            Maximum time (in seconds) to wait for the program to finish.
        debug : bool, optional
            If True, the working directory is not deleted in case of failure.

        Returns
        -------
        work_dir : str
            Path to working directory.
        '''

        work_dir = self.path_to_work(entry_id)
        os.makedirs(work_dir, exist_ok=True)

        ################################################################
        # check if program output exists
        ################################################################

        if (
            os.path.exists(self.path_to_outputs(entry_id)['couplings']) and
            os.stat(self.path_to_outputs(entry_id)['couplings']).st_size != 0 and
            os.path.exists(self.path_to_outputs(entry_id)['mapping']) and
            os.stat(self.path_to_outputs(entry_id)['mapping']).st_size != 0
        ):
            return work_dir

        ################################################################
        # BASIC INFORMATION
        ################################################################

        atoms = prody.parsePDB(pdb_path, subset='calpha')
        if self.use_monomers:
            atoms = atoms.select(f'chain {atoms.getChids()[0]}')

        n_residues_total = atoms.numAtoms()
        n_pairs = int(n_residues_total * (n_residues_total - 1) / 2)

        resnums = atoms.getResnums()
        resnames = atoms.getSequence()
        chain_ids = atoms.getChids()

        assert len(resnames) == n_residues_total
        assert len(resnums) == n_residues_total
        assert len(chain_ids) == n_residues_total

        # save mapping of residue numbers to matrix indices
        mat_idx = np.arange(n_residues_total, dtype=np.int32)
        resnum2homo_idx = {
            resnum: np.where(resnums == resnum)[0].min()
            for resnum in np.unique(resnums)
        }
        homomer_idx = [
            resnum2homo_idx[resnum] for resnum in resnums
        ]

        np.savetxt(
            self.path_to_outputs(entry_id)['mapping'],
            np.column_stack((
                mat_idx,
                resnums,
                homomer_idx,
                chain_ids,
                [c for c in resnames]
            )),
            header='idx,resnum,collapsed_idx,chain_id,resname',
            delimiter=',',
            comments='#',
            fmt='%s',
        )

        ################################################################
        # ANM ANALYSIS
        ################################################################

        anm = prody.ANM()
        anm.buildHessian(atoms, cutoff=self.cutoff)
        try:
            anm.calcModes(n_modes=self.n_modes)
        except ValueError as err:
            with open(self.failure_path, 'a+') as f:
                f.write(
                    f'{entry_id}\t'
                    f'"{err}" when calculating modes\n'
                )
            return None

        modes = anm.getEigvecs().T
        freq_sq = anm.getEigvals()

        if len(modes) != self.n_modes:
            with open(self.failure_path, 'a+') as f:
                f.write(
                    f'{entry_id}\t'
                    f'ANM computed {len(modes)} modes, expected {self.n_modes}\n'
                )
            return None

        triu_indices = np.triu_indices(n_residues_total, k=1)

        # placeholders
        header = ['res_i', 'res_j', 'resnum_i', 'resnum_j']
        data = np.hstack((
            triu_indices[0][:,None],
            triu_indices[1][:,None],
            resnums[triu_indices[0]][:,None],
            resnums[triu_indices[1]][:,None]
        ))
        fmt = ['%d', '%d', '%d', '%d']

        ### DISTANCE MAP
        header.append('contact')
        contact_mat = prody.buildDistMatrix(atoms)
        data = np.hstack((
            data,
            contact_mat[triu_indices][:,None]
        ))
        fmt.append('%f')

        ### COVARIANCE COUPLINGS
        header.append('covar')
        data = np.hstack((
            data,
            prody.calcCrossCorr(anm, norm=True, n_cpu=4)[triu_indices][:,None]
        ))
        fmt.append('%f')

        ### CODIRECTIONALITY COUPLINGS

        numerator_mat = np.zeros((n_residues_total, n_residues_total))
        for mode_idx in range(self.n_modes):

            for res1_idx in range(n_residues_total-1):
                mode_res1 = modes[mode_idx][3*res1_idx:3*res1_idx+3]
                for res2_idx in range(res1_idx+1, n_residues_total):
                    mode_res2 = modes[mode_idx][3*res2_idx:3*res2_idx+3]

                    # computation starts here

                    unit_vector1 = mode_res1 / np.linalg.norm(mode_res1)
                    unit_vector2 = mode_res2 / np.linalg.norm(mode_res2)

                    numerator_addend = np.dot(
                        unit_vector1,
                        unit_vector2
                    ) / freq_sq[mode_idx]

                    numerator_mat[res1_idx][res2_idx] += numerator_addend
                    # numerator_mat[res2_idx][res1_idx] = numerator_mat[res1_idx][res2_idx]

        codir_mat = numerator_mat / ( 1/freq_sq[:self.n_modes] ).sum()

        if np.isnan(codir_mat).any():
            with open(self.failure_path, 'a+') as f:
                f.write(
                    f'{entry_id}\t'
                    f'nan found in codir couplings\n'
                )
            return None

        header.append('codir')
        data = np.hstack((
            data,
            codir_mat[triu_indices][:,None]
        ))
        fmt.append('%f')

        ### COORDINATION COUPLINGS
        coords = prody.getCoords(atoms)

        f_sq_mat = np.zeros((n_residues_total, n_residues_total))
        for mode_idx in range(self.n_modes):

            for res1_idx in range(n_residues_total-1):
                mode_res1 = modes[mode_idx][3*res1_idx:3*res1_idx+3]
                for res2_idx in range(res1_idx+1, n_residues_total):
                    mode_res2 = modes[mode_idx][3*res2_idx:3*res2_idx+3]

                    # computation starts here
                    unit_vector = (
                        coords[res1_idx]-coords[res2_idx]
                    ) / contact_mat[res1_idx][res2_idx]

                    f_sq_addend = (np.dot(
                        unit_vector,
                        mode_res1 - mode_res2
                    ))**2 / freq_sq[mode_idx]

                    f_sq_mat[res1_idx][res2_idx] += f_sq_addend
                    # f_sq_mat[res2_idx][res1_idx] = f_sq_mat[res1_idx][res2_idx]

        coord_mat = 1 - 0.5 * np.sqrt(
            f_sq_mat # / ( 1/freq_sq[:self.n_modes] ).sum()
        )

        header.append('coord')
        data = np.hstack((
            data,
            coord_mat[triu_indices][:,None]
        ))
        fmt.append('%f')

        ### DEFORMATION COUPLINGS
        deform_mat = np.empty((n_residues_total, n_residues_total))

        hess_matrix = anm.getHessian()
        try:
            hess_pseudo = np.linalg.pinv(hess_matrix)
        except np.linalg.LinAlgError as err:
            with open(self.failure_path, 'a+') as f:
                f.write(
                    f'{entry_id}\t'
                    f'"{err}" when computing deformation\n'
                )
            return None

        for res1_idx in range(n_residues_total):
            res1_x = res1_idx * 3

            sub_matrix = hess_pseudo[res1_x:res1_x+3,res1_x:res1_x+3]

            R_ii = np.dot(sub_matrix, sub_matrix.T)
            eigenvals_ii = np.linalg.eigvals(R_ii)
            deform_ii = np.max(eigenvals_ii)

            deform_mat[res1_idx][res1_idx] = deform_ii

            for res2_idx in range(res1_idx+1, n_residues_total):
                res2_x = res2_idx * 3

                sub_matrix2 = hess_pseudo[res1_x:res1_x+3,res2_x:res2_x+3]

                R_ij = np.dot(sub_matrix2, sub_matrix2.T)
                eigenvals_ij = np.linalg.eigvals(R_ij)
                deform_ij = np.max(eigenvals_ij)

                deform_mat[res1_idx][res2_idx] = deform_ij
                # deform_mat[res2_idx][res1_idx] = deform_ij

        header.append('deform')
        data = np.hstack((
            data,
            deform_mat[triu_indices][:,None]
        ))
        fmt.append('%f')

        ################################################################
        # WRAP UP
        ################################################################

        # save data
        np.savetxt(
            self.path_to_outputs(entry_id)['couplings'],
            data,
            header=','.join(header),
            delimiter=',',
            comments='#',
            fmt=','.join(fmt)
        )

        return work_dir

    def batch_run(self, entry_ids, pdb_path_list, retry=False):

        '''Executes TNM software on a list of protein accessions.

        Calls `run` method for each accession in `accessions` list. Also
        saves a file containing accessions that failed to run.

        Parameters
        ----------
        accessions : list of str
            List of PDB accession codes.
        pdb_path_list : list of str
            List of paths to PDB files.
        retry : bool
            If True, retry failed accessions.
        debug : bool
            If True, do not delete directories containing failed entries.
            Directory removal does not work in Windows WSL (PermissionError),
            hence must be set to `True` when running on windows.
        '''

        assert len(entry_ids) == len(pdb_path_list)

        successful_entries = []
        failed_entries = []

        entries_to_skip = self.get_failed_entries()
        pbar = tqdm(entry_ids, dynamic_ncols=True, ascii=True)
        for i, entry_id in enumerate(pbar):
            # if i == 5:
                # raise Exception
            pbar.set_description(f'ANM {entry_id:<12s}')
            pdb_path = pdb_path_list[i]

            # skip entries that failed in previous runs
            if entry_id in entries_to_skip and not retry:
                failed_entries.append(entry_id)
                continue

            # keep a list of accessions successfully processed
            work_dir = self.run(
                entry_id=entry_id,
                pdb_path=pdb_path,
            )
            if work_dir is not None:
                successful_entries.append(entry_id)
            else:
                failed_entries.append(entry_id)

        # remove duplicated accessions in failure file
        np.savetxt(
            self.failure_path,
            self.get_failed_entries(return_reason=True),
            fmt='%s\t%s'
        )

        # failed accessions
        failed_entries = np.unique(failed_entries)

        return successful_entries, failed_entries

    def cleanup_failed(self):
        pass

    def get_resnames(self, entry_id, return_ids=False):
        '''Get residue names from ANM.

        Ensures that there are no missing residues in the sequence.

        Parameters
        ----------
        entry_id : str
            Entry identifier.

        Returns
        -------
        resnames : list of str (np.ndarray)
            List of residue names.
        resids : list of int (np.ndarray)
            List of residue identifiers.

        '''

        map_path = self.path_to_outputs(entry_id)['mapping']

        mapping = np.loadtxt(map_path, dtype=np.str_)

        ### get residue identifier for TNM (used in dynamic coupling
        ### files)
        dc_idx = mapping[:,0]

        # check if dc_idx are all integers
        if not all([s.isdecimal() for s in dc_idx]):
            raise ValueError('dc_idx are not all integers')
        # further check if dc_idx is sequential
        dc_idx = np.array(dc_idx, dtype=np.int32)
        if not (np.diff(dc_idx) == 1).all():
            raise ValueError('dc_idx is not sequential')

        ### get residue identifier in .pdb file (input to TNM program)
        ### also get residue type
        sp = np.char.split(mapping[:,1], sep='_')
        resnames = np.array([r[0] for r in sp])
        auth_seq_ids = [r[1] for r in sp]

        # check if auth_seq_ids are all integers
        if not all([s.isdecimal() for s in auth_seq_ids]):
            raise ValueError('auth_seq_ids are not all integers')
        # further check if auth_seq_ids is sequential
        auth_seq_ids = np.array(auth_seq_ids, dtype=np.int32)
        if not (np.diff(auth_seq_ids) == 1).all():
            raise ValueError('auth_seq_ids is not sequential')

        if return_ids:
            return resnames, dc_idx #, auth_seq_ids
        else:
            return resnames

    def get_couplings(self, entry_id, et_type):

        # Use this to get couplings after modal analysis.

        if self.cached_couplings_entry == entry_id:

            data = self.cached_couplings

        else:

            couplings_file = self.path_to_outputs(entry_id)['couplings']

            if not os.path.exists(couplings_file):
                raise FileExistsError(
                    f'Coupling file for {entry_id} does not exist.'
                )
            elif os.stat(couplings_file).st_size == 0:
                raise FileExistsError(
                    f'Coupling file for {entry_id} exists but is empty.'
                )

            data = np.loadtxt(
                couplings_file,
                delimiter=',',
                dtype=np.str_,
                comments=None
            )
            self.cached_couplings_entry = entry_id
            self.cached_couplings = data

        header = data[0]
        edge_index = data[1:, :2].astype(np.int32)
        resnum_ij = data[1:, 2:4].astype(np.int32)
        couplings = data[1:, 4:].astype(np.float32)

        if et_type not in header:
            raise ValueError(
                f'Coupling type {et_type} not found in {entry_id} file.'
            )

        et_idx = np.argwhere(header[4:] == et_type)[0][0]

        return edge_index, resnum_ij, couplings[:, et_idx]

    def get_mapping(self, entry_id):

        # Use this to get residue mapping after modal analysis.

        if self.cached_mapping_entry == entry_id:

            data = self.cached_mapping

        else:

            mapping_file = self.path_to_outputs(entry_id)['mapping']

            if not os.path.exists(mapping_file):
                raise FileExistsError(
                    f'Mapping file for {entry_id} does not exist.'
                )
            elif os.stat(mapping_file).st_size == 0:
                raise FileExistsError(
                    f'Mapping file for {entry_id} exists but is empty.'
                )

            data = np.loadtxt(
                mapping_file,
                delimiter=',',
                dtype=np.str_,
                comments=None
            )

            self.cached_mapping_entry = entry_id
            self.cached_mapping = data

        header = data[0]
        mapping = data[1:,:3].astype(np.int32)

        return mapping[:,0], mapping[:,1], mapping[:,2], data[1:,3]

class TNM_Computer:
    '''Computes TNM features for a given protein structure.

    The TNM program must be installed and accessible via command line
    (i.e. in PATH). The program is published at
    https://github.com/ugobas/tnm, and is described in
    Phys. Rev. Lett. 104, 228103 (2010).
    '''

    def __init__(
        self, n_sigmas=-10000  # output all couplings (no thresholding)
    ):

        self.tnm_dir = os.path.join(collation_dir, 'TNM')
        self.setup_name = f'run-sigma{n_sigmas}'

        failure_filename = 'tnm-failed_entries.tsv'
        self.failure_path = os.path.join(self.tnm_dir, failure_filename)

        # template_path = os.path.join(self.tnm_dir, 'template.in')
        template_path = os.path.join(
            os.path.dirname(__file__), 'template-tnm.in'
        )
        with open(template_path, 'r') as f:
            self.template_script = f.read()

        self.n_sigmas = n_sigmas

    def get_failed_entries(self, return_reason=False):
        '''Returns accessions for which TNM computation failed.'''

        if (
            os.path.exists(self.failure_path)
            and os.stat(self.failure_path).st_size != 0
        ):
            entries = np.loadtxt(
                self.failure_path, dtype=np.str_, delimiter='\t'
            ).reshape((-1, 2))
        else:
            entries = np.empty((0, 2), dtype=np.str_)

        if return_reason:
            return np.unique(entries, axis=0)
        else:
            return np.unique(entries[:, 0])

    def _path_to_work(self, accession):
        return os.path.join(self.tnm_dir, self.setup_name, accession)

    def path_to_outputs(self, accession, chains='A'):

        work_dir = self._path_to_work(accession)

        # chains = accession.split('-')[1].replace('_', '')
        prefix = f'{accession}{chains}_MIN{4.5:.1f}_ALL_PHIPSIPSI'
        filenames = {
            'mapping': f'{prefix}.names.dat',
            'cont': f'{prefix}_Cont_Mat.txt',
            'coord': f'{prefix}.coordination_coupling.dat',
            'codir': f'{prefix}.directionality_coupling.dat',
            'deform': f'{prefix}.deformation_coupling.dat',
            'bfactor': f'{prefix}.MSF.dat',
        }

        return {k: os.path.join(work_dir, filenames[k]) for k in filenames}

    def path_to_merged(self, accession):
        return os.path.join(self.tnm_dir, self.setup_name, f'{accession}.json')

    def run(
        self,
        accession,
        pdb_path,
        # timeout=60,
    ):

        '''Runs TNM program on a given protein structure.

        Parameters
        ----------
        accession : str
            PDB accession code.
        pdb_path : str
            Path to PDB file.
        timeout : int, optional
            Maximum time (in seconds) to wait for the program to finish.

        Returns
        -------
        work_dir : str
            Path to working directory.
        '''

        work_dir = self._path_to_work(accession)
        script_file = os.path.join(work_dir, 'tnm.in')
        tnm_log_file = os.path.join(work_dir, 'tnm.log')

        path_to_files = self.path_to_outputs(accession)

        ################################################################
        # check if program output exists
        ################################################################
        # check if merged file exists
        if os.path.exists(self.path_to_merged(accession)):
            return work_dir
        else:
            os.makedirs(work_dir, exist_ok=True)

        # check if execution is successful by reading log file
        if os.path.exists(tnm_log_file) and os.stat(tnm_log_file).st_size != 0:
            with open(tnm_log_file, 'r') as f:
                final_line = f.readlines()[-1]
        else:
            final_line = ''
        # check if required output files exist
        file_existence = []
        for key in path_to_files:
            path_to_file = path_to_files[key]
            file_existence.append(os.path.exists(path_to_file))

        if all(file_existence) and final_line.startswith('Total Time'):
            self.merge_output_file(accession)
            return work_dir

        ################################################################
        # run program
        ################################################################
        # modify template script
        replacements = [
            ('PDBID_PLACEHOLDER', pdb_path),
            ('CHAINID_PLACEHOLDER', 'ALL'),
            ('CUTOFF', '4.5'),
            ('SIGMA_PLACEHOLDER', str(self.n_sigmas)),
        ]
        script_content = self.template_script
        for old, new in replacements:
            script_content = script_content.replace(old, new)

        # save modified script
        with open(script_file, 'w+') as f:
            f.write(script_content)

        # change directory to `work_dir` and execute the script
        cwd = os.getcwd()
        f_log = open(tnm_log_file, 'w')
        os.chdir(work_dir)
        try:
            # tnm software must be in PATH
            subprocess.run(
                ['tnm', '-file', script_file], stdout=f_log, timeout=60
            )
        except subprocess.TimeoutExpired as err:
            # timeouts are considered failures
            os.chdir(cwd)
            f_log.close()
            shutil.rmtree(work_dir)
            with open(self.failure_path, 'a+') as f:
                f.write(
                    f'{accession}\t'
                    f'timeout (60s on {platform.node()})\n'
                )
            return None
        except Exception as err:
            # remove `work_dir` in case of other errors
            os.chdir(cwd)
            f_log.close()
            shutil.rmtree(work_dir)
            raise
        else:
            os.chdir(cwd)
            f_log.close()

        # check if execution is successful by reading log file
        with open(tnm_log_file, 'r') as f:
            final_line = f.readlines()[-1]
        if not final_line.startswith('Total Time'):
            shutil.rmtree(work_dir)
            with open(self.failure_path, 'a+') as f:
                f.write(f'{accession}\tincomplete execution\n')
            return None
        # also check if required output files exist
        file_existence = []
        for key in path_to_files:
            path_to_file = path_to_files[key]
            file_existence.append(os.path.exists(path_to_file))
        if not all(file_existence):
            shutil.rmtree(work_dir)
            with open(self.failure_path, 'a+') as f:
                f.write(f'{accession}\tmissing files\n')
            return None

        if os.path.exists(work_dir):
            self.merge_output_file(accession)
            return work_dir # does not remove entry in failure file
        else:
            with open(self.failure_path, 'a+') as f:
                f.write(f'{accession}\tmissing directory\n')
            return None

    def batch_run(
        self,
        accessions,
        pdb_path_list,
        retry=False,
        # timeout=60,
    ):

        '''Executes TNM software on a list of protein accessions.

        Calls `run` method for each accession in `accessions` list. Also
        saves a file containing accessions that failed to run.

        Parameters
        ----------
        accessions : list of str
            List of PDB accession codes.
        pdb_path_list : list of str
            List of paths to PDB files.
        retry : bool
            If True, retry failed accessions.
        '''

        successful_accessions = []
        failed_accessions = []

        accessions_to_skip = self.get_failed_entries()

        pbar = tqdm(accessions, dynamic_ncols=True, ascii=True)
        for i, accession in enumerate(pbar):
            # if i == 5:
            # raise Exception
            pbar.set_description(f'TNM {accession:<12s}')
            pdb_path = pdb_path_list[i]

            if accession in accessions_to_skip:
                # skip entries that failed in previous runs
                if not retry:
                    failed_accessions.append(accession)
                    continue
                # remove previous record from failure file
                else:
                    np.savetxt(
                        self.failure_path,
                        self.get_failed_entries(return_reason=True)[
                            accessions_to_skip!=accession
                        ],
                        fmt='%s\t%s'
                    )
                    accessions_to_skip = self.get_failed_entries()

            # keep a list of accessions successfully processed
            work_dir = self.run(
                accession=accession,
                pdb_path=pdb_path,
                # timeout=timeout,
            )
            if work_dir is not None:
                successful_accessions.append(accession)
            else:
                failed_accessions.append(accession)

        # remove duplicated accessions in failure file
        np.savetxt(
            self.failure_path,
            self.get_failed_entries(return_reason=True),
            fmt='%s\t%s'
        )

        # failed accessions
        failed_accessions = np.unique(failed_accessions)

        return successful_accessions, failed_accessions

    def merge_output_file(self, accession):
        '''Merge output files generated by TNM program.

        Parameters
        ----------
        accessions : str
            PDB accession codes.
        '''

        _path_to_work = self._path_to_work(accession)
        filenames = os.listdir(_path_to_work)

        # read all output files
        all_content = {}
        for filename in filenames:
            with open(os.path.join(_path_to_work, filename), 'r') as f:
                content = f.read()
            all_content[filename] = content

        # save merged output file
        output_dir = self.path_to_merged(accession)

        with open(output_dir, 'w+', encoding='utf-8') as f:
            json.dump(
                all_content,
                f,
                indent=4,
                sort_keys=True,
            )

        shutil.rmtree(_path_to_work)

    def cleanup_failed(self):
        '''Remove directories of failed entries. Does not work in Windows WSL.'''
        failed_accessions = self.get_failed_entries()

        removed_accessions = []
        for accession in failed_accessions:
            work_dir = self._path_to_work(accession)
            if os.path.exists(work_dir):
                removed_accessions.append(accession)
                shutil.rmtree(work_dir)

        print('# of directories removed:', len(removed_accessions))
        return removed_accessions

    def get_resnames(self, accession, return_ids=False):
        '''Get residue names from TNM output files.

        Reads `names.dat` generated by TNM program. Ensures that there
        are no missing residues in the sequence.

        Parameters
        ----------
        accession : str
            PDB accession code.

        Returns
        -------
        resnames : list of str (np.ndarray)
            List of residue names.
        resids : list of int (np.ndarray)
            List of residue identifiers.

        '''

        map_path = self.path_to_outputs(accession)['mapping']
        keyname = os.path.basename(map_path)

        # read merged output
        with open(self.path_to_merged(accession), 'r') as f:
            mapping_str = json.load(f)[keyname]

        mapping = np.loadtxt(mapping_str.split('\n'), dtype=np.str_)

        ### get residue identifier for TNM (used in dynamic coupling files)
        dc_idx = mapping[:, 0]

        # check if dc_idx are all integers
        if not all([s.isdecimal() for s in dc_idx]):
            raise ValueError('dc_idx are not all integers')
        # further check if dc_idx is sequential
        dc_idx = np.array(dc_idx, dtype=np.int32)
        if not (np.diff(dc_idx) == 1).all():
            raise ValueError('dc_idx is not sequential')

        ### get residue identifier in .pdb file (input to TNM program)
        ### also get residue type
        sp = np.char.split(mapping[:, 1], sep='_')
        resnames = np.array([r[0] for r in sp])
        auth_seq_ids = np.array([r[1] for r in sp], dtype=np.str_)
        auth_asym_id = np.array([r[2] for r in sp], dtype=np.str_)

        if return_ids:
            identifiers = {
                'dc_idx': dc_idx,
                'auth_seq_ids': auth_seq_ids,
                'auth_asym_id': auth_asym_id,
            }
            return resnames, identifiers
        else:
            return resnames

if __name__ == '__main__':

    pdb_dir = '/mnt/hdd/yenlin/data/Protein_Data_Bank/pdb-biomt'

    enm_computer = ANM_Computer(
        cutoff=12,
        n_modes=20,
        use_monomers=True,
    )

    # print(enm_computer.get_failed_entries())

    # entry_id = '1A0G-1'
    # entry_id = '4HP2-1'

    # work_dir = enm_computer.run(
    #     entry_id=entry_id,
    #     pdb_path=f'{pdb_dir}/{entry_id}.pdb',
    # )

    # if work_dir:
    #     print(f'ANM computation for {entry_id} completed')
    # else:
    #     print(f'ANM computation for {entry_id} failed')

    entry_ids = [
        # '19HC-1',
        # '1A05-1',
        '1A0G-1',
        # '1A0J-2',
        # '1A0M-1',
        # '1A12-1',
        '4HP2-1'
    ]

    successful_entries, failed_entries = enm_computer.batch_run(
        entry_ids=entry_ids,
        pdb_path_list=[f'{pdb_dir}/{entry_id}.pdb' for entry_id in entry_ids],
        retry=False
    )
    print(successful_entries)
    print(failed_entries)


    # removed_accessions = enm_computer.cleanup_failed()
    # print(removed_accessions)
    # print(len(removed_accessions))

    # # # work_dir = enm_computer.run(
    # # #     'Q6ZS30-AFv4',
    # # #     '/Users/sebastian/Dropbox/projects/ai-thermostability/code/data/external/AlphaFoldDB/pdb/Q6ZS30-AFv4.pdb'
    # # # )
    # # # print(work_dir)

    # # root = '/Users/sebastian/Dropbox/projects/ai-thermostability/code/data/external/AlphaFoldDB/pdb'
    # # accessions = ['Q6ZS30-AFv4', 'D6RIN3-AFv4', 'D6RE34-AFv4', 'E5RJZ4-AFv4', 'H3BS66-AFv4', 'Q93HR1-AFv4', 'H2L294-AFv4', 'K7EKI6-AFv4', 'E9QG37-AFv4']
    # # pdb_path_list = [os.path.join(root, accession+'.pdb') for accession in accessions]

    # # successful, failed = enm_computer.batch_run(accessions,
    # #                                          pdb_path_list,
    # #                                          retry=False)

    # # print(successful)
    # # print(failed)
