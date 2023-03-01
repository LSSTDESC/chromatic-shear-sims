"""
sagittarius -- Arrow datasets for GalSim
"""

import os
import re

import pyarrow.compute as pc
import pyarrow.dataset as ds

from galsim.catalog import Catalog
from galsim.config.input import RegisterInputType, RegisterValueType, InputLoader, GetInputObj
from galsim.config.value import SetDefaultIndex, GetAllParams, _GetBoolValue


def parse_predicate(predicate_tree):
    """Parse a predicate tree intro a pyarrow compute expression
    """
    # Parse through the tree
    for k, v in predicate_tree.items():
        f = getattr(pc, k)
        if type(v) is list:
            return f(*[parse_predicate(_v) for _v in v])
        else:
            return f(v)


def match_expression(names, expressions):
    """Match names to regular expressions
    """
    return [
        name for name in names
        for expression in expressions
        if re.match(expression, name)
    ]


class ArrowDataset(Catalog):
    """GalSim interface for Arrow datasets
    """

    _req_params = { 'dataset' : str }
    _opt_params = { 'dir' : str , 'format' : str, 'columns' : list, 'predicate' : dict }

    def __init__(self, dataset, dir=None, format=None, columns=None, predicate=None):

        print(f"Initializating dataset for {dataset}")
        # First build full file_name
        self.dataset = dataset.strip()
        self.file_name = self.dataset  # For compatibility with Catalog
        if dir is not None:
            self.file_name = os.path.join(dir, self.filename)

        # default to assuming dataset is parquet
        if format is None:
            format = "parquet"

        self.format = format
        self.file_type = self.format  # For compatibility with Catalog

        self._dataset = ds.dataset(self.dataset, format=self.format)
        self.names = self._dataset.schema.names

        if columns == []: columns = None  # Note that this will return _all_ columns in the table
        elif columns == None:
            columns = None
        else:
            columns = match_expression(self.names, columns)
        self.columns = columns
        # self.columns = \
        #     ["redshift"] \
        #     + [_q for _q in self._dataset.schema.names if re.match(r"mag_true_\w_lsst$", _q)] \
        #     + [_q for _q in self._dataset.schema.names if re.match(r"sed_\d+_\d+_no_host_extinction$", _q)]

        if predicate == {}:
            predicate = None  # Note that this will return _all_ rows in the table
        elif predicate == None:
            predicate = None
        else:
            predicate = parse_predicate(predicate)
        self.predicate = predicate

        self.comments = None  # For compatibility with Catalog
        self.hdu = None  # For compatibility with Catalog

        self._ncols = len(self.names)
        self._scanner = self._dataset.scanner(
            columns=self.columns,
            filter=self.predicate,
        )
        self.nobjects = 0  # self._scanner.count_rows()
        self._used = 0
        self._batches = self._scanner.to_batches()
        self._batch = None
        self._next_batch()

    def _next_batch(self):
        """Iterate until the next batch of non-zero size
        """
        try:
            self._batch = next(self._batches)
            # print(f"next batch!")
        except StopIteration:
            self._batches = self._scanner.to_batches()
            self._batch = next(self._batches)
            print(f"resetting batches for {self}")
        while len(self._batch) < 1:
            # print("whoops empty batch!")
            try:
                self._batch = next(self._batches)
                # print("next batch!")
            except StopIteration:
                self._batches = self._scanner.to_batches()
                self._batch = next(self._batches)
                print(f"resetting batches for {self}")
        self.nobjects = len(self._batch)
        self._used = 0
        print(f"Drawing batch of size {self.nobjects} from {self}")

    # def getRows(self, indices):
    #     return self._scanner.take(indices).to_pandas()

    def getRow(self, index):
        """Get a row from the current batch; if as many items have been taken
        from the batch as there are in the batch, get the next batch. This
        attempts to provide decent coverage of the dataset while also handling
        small batch sizes due to any predicate filtering
        """
        # Check if we need to consume the next batch
        # This should run during the first call of getRow
        if self._used >= self.nobjects:
            self._next_batch()
        # return self._scanner.take([index]).to_pydict()
        # row = self._batch.take([index]).to_pydict()
        # FIXME verify this -- we take the modulus in case a new batch is
        #       generated _after_ indices are computed by GalSim
        row = self._batch.take([index % self.nobjects]).to_pydict()  # FIXME
        self._used += 1
        return row


    def get(self, index, col):
        """Return the data for the given ``index`` and ``col`` in its native type.
        """
        if col not in self.names:
            raise GalSimKeyError("Column is not in schema for dataset %s"%self.file_name, col)
        if not isinstance(index, int):
            raise GalSimIndexError("Index must be an int for dataset %s"%self.file_name, index)
        if index < 0 or index >= self.nobjects:
            raise GalSimIndexError("Index is invalid for dataset %s"%self.file_name, index)
        # return self._scanner.take([index])[col].to_pylist().pop()
        if self._used >= self.nobjects:
            self._next_batch()
        val = self._batch.take([index % self.nobjects])[col].to_pylist().pop()  # FIXME
        self._used += 1
        return val


    def __repr__(self):
        return f"sagittarius.ArrowDataset(dataset={self.file_name}, file_type={self.file_type})"


    def __str__(self):
        return f"sagittarius.ArrowDataset(dataset={self.file_name})"


def _GenerateFromArrowDataset(config, base, value_type):
    """Return a value read from an input catalog
    """
    input_cat = GetInputObj('arrow_dataset', config, base, 'ArrowDataset')

    # Setup the indexing sequence if it hasn't been specified.
    # The normal thing with a Catalog is to just use each object in order,
    # so we don't require the user to specify that by hand.  We can do it for them.
    SetDefaultIndex(config, input_cat.getNObjects())

    req = { 'col' : str , 'index' : int }
    opt = { 'num' : int }
    kwargs, safe = GetAllParams(config, base, req=req, opt=opt)
    col = kwargs['col']
    index = kwargs['index']

    if value_type is str:
        val = input_cat.get(index, col)
    elif value_type is float:
        val = input_cat.getFloat(index, col)
    elif value_type is int:
        val = input_cat.getInt(index, col)
    else:  # value_type is bool
        val = _GetBoolValue(input_cat.get(index, col))

    #print(base['file_num'],'Catalog: col = %s, index = %s, val = %s'%(col, index, val))
    return val, safe

RegisterInputType("arrow_dataset", InputLoader(ArrowDataset, has_nobj=True))
RegisterValueType('ArrowDataset', _GenerateFromArrowDataset, [ float, int, bool, str ], input_type='arrow_dataset')
