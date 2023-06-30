from lxml.etree import Element, SubElement, tostring, XMLParser
from xml.etree import ElementTree
from imported_utils import *
from sumo_utils.general import *

class E(list):
    """
    Builder for lxml.etree.Element
    """
    xsi = Path('F') / 'resources' / 'xml' / 'XMLSchema-instance' # http://www.w3.org/2001/XMLSchema-instance
    root_args = dict(nsmap=dict(xsi=xsi))
    def __init__(self, _name, *args, **kwargs):
        assert all(isinstance(a, E) for a in args)
        super().__init__(args)
        self._dict = kwargs
        self._name = _name

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def __getitem__(self, k):
        if isinstance(k, (int, slice)):
            return super().__getitem__(k)
        return self._dict.__getitem__(k)

    def __setitem__(self, k, v):
        if isinstance(k, (int, slice)):
            return super().__setitem__(k, v)
        return self._dict.__setitem__(k, v)

    def __getattr__(self, k):
        if k == '__array_struct__':
            raise RuntimeError('Cannot make numpy arrays with E as elements (since E subclasses list?)')
        if k in ['_dict', '_name']:
            return self.__dict__[k]
        else:
            return self[k]

    def __setattr__(self, k, v):
        if k in ['_dict', '_name']:
            self.__dict__[k] = v
        else:
            self[k] = v

    def __repr__(self):
        return self.to_string().decode()

    def children(self, tag):
        return [x for x in self if x._name == tag]

    @classmethod
    def from_element(cls, e):
        return E(e.tag, *(cls.from_element(x) for x in e), **e.attrib)

    @classmethod
    def from_path(cls, p):
        return cls.from_element(ElementTree.parse(p, parser=XMLParser(recover=True)).getroot())

    @classmethod
    def from_string(cls, s):
        return cls.from_element(ElementTree.fromstring(s))


class XML_Decoder():

    def __init__(self, file_path):
        self.tree = ElementTree.parse(file_path)
        # get root element
        self.root = self.tree.getroot()

    def decode_routes(self):
        vehicle_ids = []
        for child in self.root:
            if child.tag == "vehicle":
                vehicle_ids.append(child.attrib['id'])
        return vehicle_ids    

    def decode_tl(self):
        tl_ids = []
        for child in self.root:
            if child.tag == "junction":
                if child.attrib['type'] == "traffic_light":
                    tl_ids.append(child.attrib['id'])
        return tl_ids    