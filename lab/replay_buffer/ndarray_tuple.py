# Adapted from: https://github.com/astooke/rlpyt/blob/master/rlpyt/utils/collections.py
from collections import namedtuple
from typing import List, Tuple


def ndarray_tuple(name: str, field_names: List):
    NamedTuple = namedtuple(name, field_names)

    def __getitem__(self, idx):
        """Returns new ndarray_tuple with all fields indexed at idx."""
        try:
            return type(self)(*(None if field is None else field[idx] for field in self))
        except IndexError:
            for i, field in enumerate(self):
                if field is None:
                    continue
                try:
                    field[idx]
                except IndexError as e:
                    message = f"Occurred in ndarray_tuple '{self.__name__}' at field '{self._fields[i]}' ("
                    raise IndexError(message + str(e) + ").")

    def __setitem__(self, idx, value):
        """Writes ndarray_tuple with identical fields to position idx."""
        if set(value.keys()) != set(self.keys()):
            raise KeyError(f"Keys of inserted value {value.keys()} do not match the ndarray_tuple keys {self.keys()}.")
        for j, (s, v) in enumerate(zip(self, value)):
            s[idx] = v

    def __contains__(self, key):
        return key in self._fields

    def items(self):
        for key, value in zip(self._fields, self):
            yield key, value

    def keys(self):
        keys = []
        for key in self._fields:
            keys.append(key)
        return keys

    @property
    def shape(self):
        """Returns dict of shapes for the ndarray_tuple fields."""
        shapes = {}
        for key, value in self.items():
            try:
                shapes[key] = value.shape
            except AttributeError:
                print(f"Cannot determine shape at field '{key}', since value is not np.ndarray.")
                shapes[key] = None
        return shapes

    @property
    def dtype(self):
        """Returns dict of dtypes for the ndarray_tuple fields."""
        dtypes = {}
        for key, value in self.items():
            try:
                dtypes[key] = value.dtype
            except AttributeError:
                print(f"Cannot determine dtype at field '{key}', since value is not np.ndarray.")
                dtypes[key] = None
        return dtypes

    def reshape(self, shape) -> ndarray_tuple:
        """Reshape arrays either to one given shape, or list of shapes."""
        if isinstance(shape, Tuple):
            return type(self)(*(None if field is None else field.reshape(shape) for field in self))
        elif isinstance(shape, List):
            fields = []
            for idx, field in enumerate(self):
                assert isinstance(shape[idx], Tuple), f"Given shape {shape[idx]} could not be interpreted as a tuple."
                fields.append(field.reshape(shape[idx]))
            return type(self)(*fields)

    def replace(self, field_name: str, new_value) -> ndarray_tuple:
        """Replaces the value of a given field."""
        fields = []
        assert field_name in self, f"Field name '{field_name}' not in ndarray_tuple fields {self.keys()}."
        for name, value in self.items():
            if name == field_name:
                fields.append(new_value)
            else:
                fields.append(value)
        return type(self)(*fields)

    namespace_dict = {
        "__name__": name,
        "__getitem__": __getitem__,
        "__setitem__": __setitem__,
        "__contains__": __contains__,
        "items": items,
        "keys": keys,
        "shape": shape,
        "dtype": dtype,
        "reshape": reshape,
        "replace": replace,
    }
    bases = (NamedTuple,)
    NdarrayTuple = type(name, bases, namespace_dict)
    return NdarrayTuple
