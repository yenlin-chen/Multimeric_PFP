if __name__ in ['__main__', 'retrievers']:
    from __init__ import external_dir
else:
    from .__init__ import external_dir

import os, requests
import os.path as osp
import numpy as np
from tqdm import tqdm

class AlphaFold_Retriever: # The Golden AF Retriever

    '''Class for retrieving protein structures from AlphaFoldDB.

    This class is for managing cache and log for AlphaFoldDB retrievals,
    and should only be instantiated once in every experiment / data
    processing procedure to ensure correct caching and logging.

    Note on accession number modification:
    The function expects the UniProt accession number as input,
    and will modify the accession number to project-internal accessions
    only when saving the downloaded structure to
    `data/external/AlphaFold/pdb`. The UniProt accession number is used
    for reporting both failed and successful retrievals (returned value
    for functions and lists saved).

    Sequences with length over 2,700 are not available in AlphaFoldDB
    (whole proteome download is required), hence not retrievable using
    this class.

    Parameters
    ----------
    version : str
        Version of AlphaFoldDB to use. Default is 'v4'.
    '''

    def __init__(self, version=4):

        self.v = version
        self.af_dir = osp.join(external_dir, 'AlphaFoldDB')

        failure_filename = 'alphafold-unavailable_entry.tsv'
        self.failure_path = osp.join(self.af_dir, failure_filename)
        self.failed_accessions = self.get_failed_accessions(return_reason=True)

        self.base_url = 'https://alphafold.ebi.ac.uk/files'

    def get_failed_accessions(self, return_reason=False):
        if (osp.exists(self.failure_path) and
            os.stat(self.failure_path).st_size!=0):
            entries = np.loadtxt(self.failure_path,
                                 dtype=np.str_,
                                 delimiter='\t').reshape((-1,3))
        else:
            entries = np.empty((0,3), dtype=np.str_)

        if return_reason:
            return entries
        else:
            return np.unique(entries[:,0])

    def modify_accession(self, accession):
        '''Modify UniProt accession number to project-internal accessions'''

        suffix = f'-AFv{self.v}'
        if suffix in accession:
            raise ValueError('accession is already modified')
        if isinstance(accession, str):
            return accession+suffix
        elif isinstance(accession, np.ndarray):
            return np.char.add(accession, suffix)
        elif isinstance(accession, list):
            return [acc+suffix for acc in accession]
        else:
            raise TypeError('accession must be str, np.ndarray, or list')

    def unmodify_accession(self, accession):
        '''Revert project-internal accessions to UniProt accession number'''

        suffix = f'-AFv{self.v}'
        if suffix not in accession:
            raise ValueError('accession is already unmodified')
        if isinstance(accession, str):
            return accession.replace(suffix, '')
        elif isinstance(accession, np.ndarray):
            return np.char.replace(accession, suffix, '')
        elif isinstance(accession, list):
            return [acc.replace(suffix, '') for acc in accession]
        else:
            raise TypeError('accession must be str, np.ndarray, or list')

    def path_to_file(self, accession, item_type='pdb'):
        '''Returns path to AlphaFoldDB file for a given accession.

        Parameters
        ----------
        accession : str
            Accession to retrieve.
        item_type : str
            Type of file to retrieve. Must be either 'pdb' or 'pae'.
            Default is 'pdb'.

        Returns
        -------
        str
            Path to specified file.
        '''

        if item_type not in ['pdb', 'pae']:
            raise ValueError('item_type must be either "pdb" or "pae"')

        extension = 'pdb' if item_type == 'pdb' else 'json'

        if accession.endswith(f'-AFv{self.v}'):
            return osp.join(self.af_dir, item_type, f'{accession}.{extension}')
        else:
            return osp.join(self.af_dir, item_type,
                            f'{self.modify_accession(accession)}.{extension}')

    def retrieve(self, accession, item_type):
        '''Downloads AlphaFold structures for a single accession.

        Downloads specified file from AlphaFoldDB if it cannot be found
        in local system, and saves the downloaded .pdb once downloaded.
        This function does not check if the file failed in previous
        runs.

        Parameters
        ----------
        accession : str
            Accession to retrieve.
        item_type : str
            Type of file to retrieve. Must be either 'pdb' or 'pae'.

        Returns
        -------
        str
            Path to structure file (.pdb) or PAE file (.json).
        '''

        if item_type not in ['pdb', 'pae']:
            raise ValueError('item_type must be either "pdb" or "pae"')

        # define save location
        item_path = self.path_to_file(accession, item_type=item_type)

        # check if file exists and is non-empty
        if osp.exists(item_path) and os.stat(item_path).st_size != 0:
            return item_path

        # access AlphaFoldDB since .pdb does not exist in local fs
        if item_type == 'pdb':
            url = (f'{self.base_url}/AF-{accession}-F1-'
                   f'model_v{self.v}.pdb')
        else:
            url = (f'{self.base_url}/AF-{accession}-F1-'
                   f'predicted_aligned_error_v{self.v}.json')
        response = requests.get(url)

        # operation not successful (e.g. accession not found)
        status = response.status_code
        if status != 200:
            self.failed_accessions = np.vstack(
                (self.failed_accessions, [accession, item_type, status])
            )
            return None

        # operation successful, write to cache
        try: # removes unfinished cache files
            with open(item_path, 'wb+') as c:
                c.write(response.content)
        except Exception as err:
            if osp.exists(item_path):
                os.remove(item_path)
            raise

        if osp.exists(item_path):
            return item_path
        else:
            self.failed_accessions = np.vstack(
                (self.failed_accessions, [accession, item_type, 'unknown'])
            )
            return None

    def batch_retrieve(self, accessions, item_list=['pdb', 'pae'],
                       retry=False):
        '''Retrieves AlphaFold structure and/or PAE for a list of accessions.

        Checks if files for an accession failed to download before
        calling the `retrieve()` method. Failed accessions are logged
        for reference for later runs.

        Parameters
        ----------
        accessions : list of str
            List of accessions to retrieve.
        item_list : list of str, optional
            List of items to retrieve. Items must be either 'pdb' or 'pae'.
        retry : bool, optional
            If True, will retry accessions that failed in previous runs.
            Default is False.

        Return
        ------
        successful_accessions : list of str
            List of accessions successfully retrieved.
        '''

        if not isinstance(item_list, list):
            raise TypeError('item_list must be a list of strings')
        if not all([item in ['pdb', 'pae'] for item in item_list]):
            raise ValueError('item_list must only contain "pdb" or "pae"')

        # ensure order of retrieval
        # related to how raw_dir is implemented in `datasets.py`
        item_list_ordered = []
        if 'pdb' in item_list:
            item_list_ordered.append('pdb')
        if 'pae' in item_list:
            item_list_ordered.append('pae')

        successful_accessions = []
        failed_accessions = []

        # keep track of any new failures
        new_failure = False

        pbar = tqdm(accessions, dynamic_ncols=True, ascii=True)
        for accession in pbar:
            pbar.set_description(f'{accession:<12s}')

            # skip entries that failed in previous runs
            if accession in self.failed_accessions[:,0] and not retry:
                failed_accessions.append(accession)
                continue

            # keep a list of accessions successfully processed
            success = True
            for item_type in item_list_ordered:
                item_path = self.retrieve(accession, item_type=item_type)
                if item_path is None:
                    success = False
                    break
            if success:
                successful_accessions.append(accession)
            else:
                new_failure = True
                failed_accessions.append(accession)

        # save list of failed accessions if new failures are added
        if new_failure:
            self.failed_accessions = np.unique(self.failed_accessions, axis=0)
            print(self.failure_path)
            print(self.failed_accessions)
            np.savetxt(self.failure_path, self.failed_accessions, fmt='%s\t%s\t%s')

        # failed accessions
        failed_accessions = np.unique(failed_accessions)

        return successful_accessions, failed_accessions

class UniProt_Retriever:

    def __init__(self):

        self.uniprot_dir = osp.join(external_dir, 'UniProt')

        failure_filename = 'uniprotkb-unavailable_accessions.tsv'
        self.failure_path = osp.join(self.uniprot_dir, failure_filename)
        self.failed_accessions = self.get_failed_accessions(return_reason=True)

        self.base_url = 'https://rest.uniprot.org/uniprotkb'

    def get_failed_accessions(self, return_reason=False):
        if (osp.exists(self.failure_path) and
            os.stat(self.failure_path).st_size!=0):
            entries = np.loadtxt(self.failure_path,
                                 dtype=np.str_,
                                 delimiter='\t').reshape((-1,2))
        else:
            entries = np.empty((0,2), dtype=np.str_)

        if return_reason:
            return entries
        else:
            return entries[:,0]

    def path_to_data(self, accession):
        return osp.join(self.uniprot_dir, 'kb', f'{accession}.txt')

    def retrieve(self, accession):

        # define save location
        save_path = self.path_to_data(accession)

        # check if file exists and is non-empty
        if osp.exists(save_path) and os.stat(save_path).st_size != 0:
            return save_path

        # access UniProtKB since .txt does not exist in local fs
        url = f'{self.base_url}/{accession}.txt'
        response = requests.get(url)

        # operation not successful (e.g. accession not found)
        status = response.status_code
        if status != 200:
            self.failed_accessions = np.vstack(
                (self.failed_accessions, [accession, status])
            )
            return None
        # operation successful but file is empty
        # likely due to entry being merged with some other accession
        if response.content == b'':
            self.failed_accessions = np.vstack(
                (self.failed_accessions, [accession, 'empty'])
            )
            # remove empty file
            if osp.exists(save_path):
                os.remove(save_path)
            return None

        # operation successful, write to cache
        try: # removes unfinished cache files
            with open(save_path, 'wb+') as c:
                c.write(response.content)
        except Exception as err:
            if osp.exists(save_path):
                os.remove(save_path)
            raise

        if osp.exists(save_path):
            return save_path
        else:
            self.failed_accessions = np.vstack(
                (self.failed_accessions, [accession, 'unknown'])
            )
            return None

    def batch_retrieve(self, accessions, retry=False):

        successful_accessions = []
        failed_accessions = []

        # keep track of any new failures
        new_failure = False

        pbar = tqdm(accessions, dynamic_ncols=True, ascii=True)
        for accession in pbar:
            pbar.set_description(f'{accession:<12s}')

            # skip entries that failed in previous runs
            if accession in self.failed_accessions[:,0] and not retry:
                failed_accessions.append(accession)
                continue

            # keep a list of accessions successfully processed
            data_path = self.retrieve(accession)
            if data_path is not None:
                successful_accessions.append(accession)
            else:
                new_failure = True
                failed_accessions.append(accession)

        # save list of failed accessions if new failures are added
        if new_failure:
            self.failed_accessions = np.unique(self.failed_accessions, axis=0)
            np.savetxt(self.failure_path, self.failed_accessions, fmt='%s\t%s')

        # failed accessions
        failed_accessions = np.unique(failed_accessions)

        return successful_accessions, failed_accessions

if __name__ == '__main__':

    from __init__ import deepstabp_col

    file_path = osp.join(deepstabp_col, 'cell-human.csv')

    accessions = np.loadtxt(
        file_path, usecols=0, skiprows=1, dtype=np.str_, delimiter=','
    )
    print(accessions.size)

    retr = AlphaFold_Retriever()

    retr.batch_retrieve(accessions, item_list=['pae'])
