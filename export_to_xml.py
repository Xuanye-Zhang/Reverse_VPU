"""
Export model structure to XML file
"""

from dataclasses import dataclass
from typing import Dict

from lxml import etree as ET
import numpy as np

@dataclass
class Layer:
    """
    Layer class.
    """
    layer_id: int = -1
    port_id: int = -1

    def is_valid(self) -> bool:
        """
        Checks if layer info is valid.
        """
        return self.layer_id >= 0 and self.port_id >= 0

@dataclass
class Edge:
    """
    Edge class.
    """
    l_from: Layer = Layer()
    l_to: Layer = Layer()


    def is_valid(self) -> bool:
        """
        Checks if edge info is valid.
        """
        return self.l_from and self.l_to and self.l_from.is_valid() and self.l_to.is_valid()

class XMLExporter:
    """
    XML Exporter.
    """

    def __init__(self, net_name: str, layer_names: Dict[str, str]):
        self.layer_names = layer_names
        self.layer_names['sink'] = self.layer_names['output']+'/sink'
        self.net = ET.Element('net')
        self.net.set('name', net_name)
        self.net.set('version', '10')
        self.layers = ET.SubElement(self.net, 'layers')
        self.edges = ET.SubElement(self.net, 'edges')
        self.metadata = ET.SubElement(self.net, 'meta_data')

        self.layer_id = 0
        self.layer_ports = 0

    def add_layer(self, layer):
        """
        Adds a new layer to the XML document.
        Returns:
            - new layer's ID
            - port mapping
        """
        ports = { 'input': [], 'output': [] }
        new_layer = ET.SubElement(self.layers, 'layer')
        new_layer.set('id', str(self.layer_id))
        self.layer_id += 1

        # randomize name to prevent duplicate entries
        if layer['name'] not in self.layer_names.values():
            layer['name'] = layer['name'] + '_' + str(np.random.randint(1000, 10000))
        new_layer.set('name', layer['name'])
        new_layer.set('type', layer['type'])
        new_layer.set('version', layer['version'])

        if len(layer['data'].items()) > 0:
            data = ET.SubElement(new_layer, 'data')
            for key, value in layer['data'].items():
                data.set(key, value)

        current_port = 0
        if len(layer['inputs']) > 0:
            inputs = ET.SubElement(new_layer, 'input')
            for in_dimension in layer['inputs']:
                i_port = ET.SubElement(inputs, 'port')
                i_port.set('id', str(current_port))
                i_port.set('precision', layer['out_prec'])
                ports['input'].append(current_port)
                current_port += 1
                for dim in in_dimension:
                    ET.SubElement(i_port, 'dim').text = str(dim)

        if len(layer['outputs']) > 0:
            outputs = ET.SubElement(new_layer, 'output')
            for out_dimension in layer['outputs']:
                o_port = ET.SubElement(outputs, 'port')
                o_port.set('id', str(current_port))
                o_port.set('precision', layer['out_prec'])
                ports['output'].append(current_port)
                current_port += 1
                for dim in out_dimension:
                    ET.SubElement(o_port, 'dim').text = str(dim)
        self.layer_ports = len(layer['inputs']) + len(layer['outputs'])
        return self.layer_id - 1, ports

    def add_edge(self, edge: Edge):
        """
        Adds a new edge to the XML document.
        """
        new_edge = ET.SubElement(self.edges, 'edge')
        new_edge.set('from-layer', str(edge.l_from.layer_id))
        new_edge.set('from-port', str(edge.l_from.port_id))
        new_edge.set('to-layer', str(edge.l_to.layer_id))
        new_edge.set('to-port', str(edge.l_to.port_id))

    def write_to_file(self, filename='dump.xml'):
        """
        Writes generated XML document to output file.
        """
        ET.indent(self.net, space='\t')
        xmldoc = ET.tostring(
            self.net,
            encoding='utf8',
            pretty_print=True,
            xml_declaration=False,
            doctype='<?xml version="1.0" ?>').decode()
        with open(filename, "w", encoding='utf8') as file:
            file.write(xmldoc)


add_transpose = lambda name, inshape, outshape: {
    'name': name,
    'type': 'Transpose',
    'version': 'opset1',
    'data': {},
    'inputs': inshape,
    'out_prec': 'FP16',
    'outputs': outshape
}
add_const = lambda name, data, shape, prec: {
    'name': name,
    'type': 'Const',
    'version': 'opset1',
    'data': data,
    'inputs': [],
    'out_prec': prec,
    'outputs': shape
}
add_softmax = lambda name, shape: {
    'name': name,
    'type': 'SoftMax',
    'version': 'opset1',
    'data': {'axis': '1'},
    'inputs': [shape],
    'out_prec': 'FP16',
    'outputs': [shape]
}
add_relu = lambda name, shape: {
    'name': name,
    'type': 'ReLU',
    'version': 'opset1',
    'data': {},
    'inputs': [shape],
    'out_prec': 'FP16',
    'outputs': [shape]
}
add_pooling = lambda name, data, inshape, outshape: {
    'name': name,
    'type': 'MaxPool',
    'version': 'opset1',
    'data': data,
    'inputs': [inshape],
    'out_prec': 'FP16',
    'outputs': [outshape]
}
add_avg_pooling = lambda name, data, inshape, outshape: {
    'name': name,
    'type': 'AvgPool',
    'version': 'opset1',
    'data': data,
    'inputs': [inshape],
    'out_prec': 'FP16',
    'outputs': [outshape]
}
add_addition = lambda name, shape, add_shape={}: {
    'name': name,
    'type': 'Add',
    'version': 'opset1',
    'data': {'auto_broadcast': 'numpy'},
    'inputs': [shape, add_shape],
    'out_prec': 'FP16',
    'outputs': [shape]
}
add_conv = lambda name, inshape, weightshape, outshape, strides, mnist_special_treatment = False: {
    'name': name,
    'type': 'Convolution',
    'version': 'opset1',
    'data': {
        'strides': strides,
        'dilations': '1, 1',
        'pads_begin': '0, 0',
        'pads_end': '0, 0',
        # THE NEXT LINE IS VERY IMPORTANT! Without 'same_upper', the mnist model's first conv layer results in an output shape of (1,32,24,24) instead of (1,32,28,28)!
        # This then breaks at some later stage and is really, really nasty to debug!
        'auto_pad': 'same_upper' if mnist_special_treatment else 'explicit' # seems like the MNIST model wants 'same_upper' here, but all other models take 'explicit'
        #'output_padding': '0,0', # seems this is also only for MNIST. the other models don't use this
    },
    'inputs': [inshape, weightshape],
    'out_prec': 'FP16',
    'outputs': [outshape]
}
add_dense = lambda name, inshape, weightshape, outshape: {
    'name': name,
    'type': 'MatMul',
    'version': 'opset1',
    'data': {
        'transpose_a': 'false',
        'transpose_b': 'true'
    },
    'inputs': [inshape, weightshape],
    'out_prec': 'FP16',
    'outputs': [outshape]
}
add_reshape = lambda name, inshape, indexshape, outshape: {
    'name': name,
    'type': 'Reshape',
    'version': 'opset1',
    'data': { 'special_zero': 'False' },
    'inputs': [inshape, indexshape],
    'out_prec': 'FP16',
    'outputs': [outshape]
}
