# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module contains converters to/from protobuf messages.

For example <compiler_gym.servie.proto.ActionSpace>/<compiler_gym.servie.proto.ObservationSpace> <-> <compiler_gym.spaces>,
or <compiler_gym.servie.proto.Event> <-> actions/observation.

When defining new environments <compiler_gym.service.proto.py_convertes.make_message_default_converter>
and <compiler_gym.service.proto.py_convertes.to_event_message_default_converter>
can be used as a starting point for custom converters.
"""
import json
from builtins import getattr
from typing import Any, Callable
from typing import Dict as DictType
from typing import List, Type, Union

import google.protobuf.any_pb2 as any_pb2
import networkx as nx
import numpy as np
from google.protobuf.message import Message
from gym.spaces import Space as GymSpace

from compiler_gym.service.proto.compiler_gym_service_pb2 import (
    ActionSpace,
    BooleanBox,
    BooleanRange,
    BooleanSequenceSpace,
    BooleanTensor,
    ByteBox,
    ByteSequenceSpace,
    BytesSequenceSpace,
    ByteTensor,
    CommandlineSpace,
    DictEvent,
    DictSpace,
    DiscreteSpace,
    DoubleBox,
    DoubleRange,
    DoubleSequenceSpace,
    DoubleTensor,
    Event,
    FloatBox,
    FloatRange,
    FloatSequenceSpace,
    FloatTensor,
    Int64Box,
    Int64Range,
    Int64SequenceSpace,
    Int64Tensor,
    ListEvent,
    ListSpace,
    NamedDiscreteSpace,
    ObservationSpace,
    Opaque,
    Space,
    StringSequenceSpace,
    StringSpace,
    StringTensor,
)
from compiler_gym.spaces.box import Box
from compiler_gym.spaces.commandline import Commandline, CommandlineFlag
from compiler_gym.spaces.dict import Dict
from compiler_gym.spaces.discrete import Discrete
from compiler_gym.spaces.named_discrete import NamedDiscrete
from compiler_gym.spaces.scalar import Scalar
from compiler_gym.spaces.sequence import Sequence
from compiler_gym.spaces.tuple import Tuple


def proto_to_action_space(space: ActionSpace):
    return message_default_converter(space)


class TypeBasedConverter:
    """Converter that dispatches based on the exact type of the parameter.

    >>> converter = TypeBasedConverter({ int: lambda x: float(x)})
    >>> val: float = converter(5)
    """

    conversion_map: DictType[Type, Callable[[Any], Any]]

    def __init__(self, conversion_map: DictType[Type, Callable[[Any], Any]] = None):
        self.conversion_map = {} if conversion_map is None else conversion_map

    def __call__(self, val: Any) -> Any:
        return self.conversion_map[type(val)](val)


proto_type_to_dtype_map = {
    BooleanTensor: bool,
    ByteTensor: np.int8,
    Int64Tensor: np.int64,
    FloatTensor: np.float32,
    DoubleTensor: np.float64,
    StringTensor: object,
    BooleanBox: bool,
    ByteBox: np.int8,
    Int64Box: np.int64,
    FloatBox: np.float32,
    DoubleBox: float,
    BooleanRange: bool,
    Int64Range: np.int64,
    FloatRange: np.float32,
    DoubleRange: float,
    BooleanSequenceSpace: bool,
    BytesSequenceSpace: bytes,
    ByteSequenceSpace: np.int8,
    Int64SequenceSpace: np.int64,
    FloatSequenceSpace: np.float32,
    DoubleSequenceSpace: float,
    StringSpace: str,
}


def convert_standard_tensor_message_to_numpy(
    tensor: Union[BooleanTensor, Int64Tensor, FloatTensor, DoubleTensor, StringTensor]
):
    res = np.array(tensor.value, dtype=proto_type_to_dtype_map[type(tensor)])
    res = res.reshape(tensor.shape)
    return res


def convert_numpy_to_boolean_tensor_message(tensor: np.ndarray):
    return BooleanTensor(value=tensor.flatten().tolist(), shape=tensor.shape)


def convert_byte_tensor_message_to_numpy(tensor: ByteTensor):
    res = np.frombuffer(tensor.value, dtype=np.byte)
    res = res.reshape(tensor.shape)
    return res


def convert_numpy_to_byte_tensor_message(tensor: np.ndarray):
    return ByteTensor(value=tensor.tobytes(), shape=tensor.shape)


def convert_numpy_to_int64_tensor_message(tensor: np.ndarray):
    return Int64Tensor(value=tensor.flatten(), shape=tensor.shape)


def convert_numpy_to_float_tensor_message(tensor: np.ndarray):
    return FloatTensor(value=tensor.flatten(), shape=tensor.shape)


def convert_numpy_to_double_tensor_message(tensor: np.ndarray):
    return DoubleTensor(value=tensor.flatten(), shape=tensor.shape)


def convert_numpy_to_string_tensor_message(tensor: np.ndarray):
    return StringTensor(value=tensor.flatten(), shape=tensor.shape)


convert_tensor_message_to_numpy = TypeBasedConverter(
    conversion_map={
        BooleanTensor: convert_standard_tensor_message_to_numpy,
        ByteTensor: convert_byte_tensor_message_to_numpy,
        Int64Tensor: convert_standard_tensor_message_to_numpy,
        FloatTensor: convert_standard_tensor_message_to_numpy,
        DoubleTensor: convert_standard_tensor_message_to_numpy,
        StringTensor: convert_standard_tensor_message_to_numpy,
    }
)


def convert_bytes_to_numpy(arr: bytes) -> np.ndarray:
    return np.frombuffer(arr, dtype=np.int8)


class NumpyToTensorMessageConverter:
    dtype_conversion_map: DictType[Type, Callable[[Any], Message]]

    def __init__(self):
        self.dtype_conversion_map = {
            np.bool_: convert_numpy_to_boolean_tensor_message,
            np.int8: convert_numpy_to_byte_tensor_message,
            np.int64: convert_numpy_to_int64_tensor_message,
            np.float32: convert_numpy_to_float_tensor_message,
            np.float64: convert_numpy_to_double_tensor_message,
            np.dtype(object): convert_numpy_to_string_tensor_message,
        }

    def __call__(
        self, tensor: np.ndarray
    ) -> Union[
        BooleanTensor, ByteTensor, Int64Tensor, FloatTensor, DoubleTensor, StringTensor
    ]:
        return self.dtype_conversion_map[tensor.dtype.type](tensor)


convert_numpy_to_tensor_message = NumpyToTensorMessageConverter()


def convert_trivial(val: Any):
    return val


class FromMessageConverter:
    """Convert a protobuf message to an object.

    The conversion function is chosen based on the message descriptor.
    """

    conversion_map: DictType[str, Callable[[Message], Any]]

    def __init__(self, conversion_map: DictType[str, Callable[[Message], Any]] = None):
        self.conversion_map = {} if conversion_map is None else conversion_map

    def __call__(self, message: Message) -> Any:
        return self.conversion_map[message.DESCRIPTOR.full_name](message)


class EventMessageConverter:
    message_converter: TypeBasedConverter

    def __init__(self, message_converter: TypeBasedConverter):
        self.message_converter = message_converter

    def __call__(self, event: Event):
        field = event.WhichOneof("value")
        if field is None:
            return None

        return self.message_converter(getattr(event, field))


class ToEventMessageConverter:
    converter: TypeBasedConverter
    type_field_map: DictType[Type, str]

    def __init__(self, converter: TypeBasedConverter):
        self.converter = converter
        self.type_field_map = {
            ListEvent: "event_list",
            DictEvent: "event_dict",
            bool: "boolean_value",
            int: "int64_value",
            np.float32: "float_value",
            float: "double_value",
            str: "string_value",
            BooleanTensor: "boolean_tensor",
            ByteTensor: "byte_tensor",
            Int64Tensor: "int64_tensor",
            FloatTensor: "float_tensor",
            DoubleTensor: "double_tensor",
            StringTensor: "string_tensor",
            any_pb2.Any: "any_value",
        }

    def __call__(self, val: Any) -> Event:
        converted_val = self.converter(val)
        res = Event()
        if isinstance(converted_val, Message):
            getattr(res, self.type_field_map[type(converted_val)]).CopyFrom(
                converted_val
            )
        else:
            setattr(res, self.type_field_map[type(converted_val)], converted_val)
        return res


class ListEventMessageConverter:
    event_message_converter: EventMessageConverter

    def __init__(self, event_message_converter: EventMessageConverter):
        self.event_message_converter = event_message_converter

    def __call__(self, list_event: ListEvent) -> List[Any]:
        return [self.event_message_converter(event) for event in list_event.event]


class ToListEventMessageConverter:
    to_event_converter: ToEventMessageConverter

    def __init__(self, to_event_converter: ToEventMessageConverter):
        self.to_event_converter = to_event_converter

    def __call__(self, event_list: List) -> ListEvent:
        return ListEvent(event=[self.to_event_converter(event) for event in event_list])


class DictEventMessageConverter:
    event_message_converter: EventMessageConverter

    def __init__(self, event_message_converter: EventMessageConverter):
        self.event_message_converter = event_message_converter

    def __call__(self, dict_event: DictEvent) -> DictType[str, Any]:
        return {
            key: self.event_message_converter(event)
            for key, event in dict_event.event.items()
        }


class ToDictEventMessageConverter:
    to_event_converter: ToEventMessageConverter

    def __init__(self, to_event_converter: ToEventMessageConverter):
        self.to_event_converter = to_event_converter

    def __call__(self, d: DictType) -> DictEvent:
        return DictEvent(
            event={key: self.to_event_converter(val) for key, val in d.items()}
        )


class ProtobufAnyUnpacker:
    # message type string to message class map
    type_str_to_class_map: DictType[str, Type]

    def __init__(self, type_str_to_class_map: DictType[str, Type] = None):
        self.type_str_to_class_map = (
            {
                "compiler_gym.Opaque": Opaque,
                "compiler_gym.CommandlineSpace": CommandlineSpace,
            }
            if type_str_to_class_map is None
            else type_str_to_class_map
        )

    def __call__(self, msg: any_pb2.Any) -> Message:
        message_cls = self.type_str_to_class_map[msg.TypeName()]
        unpacked_message = message_cls()
        status = msg.Unpack(unpacked_message)
        if not status:
            raise ValueError(
                f'Failed unpacking prtobuf Any message with type url "{msg.TypeName()}".'
            )
        return unpacked_message


class ProtobufAnyConverter:
    unpacker: ProtobufAnyUnpacker
    message_converter: TypeBasedConverter

    def __init__(
        self, unpacker: ProtobufAnyUnpacker, message_converter: TypeBasedConverter
    ):
        self.unpacker = unpacker
        self.message_converter = message_converter

    def __call__(self, msg: any_pb2.Any) -> Any:
        unpacked_message = self.unpacker(msg)
        return self.message_converter(unpacked_message)


class ActionSpaceMessageConverter:
    message_converter: Callable[[Any], Any]

    def __init__(self, message_converter: Callable[[Any], Any]):
        self.message_converter = message_converter

    def __call__(self, message: ActionSpace) -> GymSpace:
        res = self.message_converter(message.space)
        res.name = message.name
        return res


class ObservationSpaceMessageConverter:
    message_converter: Callable[[Any], Any]

    def __init__(self, message_converter: Callable[[Any], Any]):
        self.message_converter = message_converter

    def __call__(self, message: ObservationSpace) -> GymSpace:
        res = self.message_converter(message.space)
        res.name = message.name
        return res


def make_message_default_converter() -> TypeBasedConverter:
    conversion_map = {
        bool: convert_trivial,
        int: convert_trivial,
        float: convert_trivial,
        str: convert_trivial,
        bytes: convert_bytes_to_numpy,
        BooleanTensor: convert_tensor_message_to_numpy,
        ByteTensor: convert_tensor_message_to_numpy,
        Int64Tensor: convert_tensor_message_to_numpy,
        FloatTensor: convert_tensor_message_to_numpy,
        DoubleTensor: convert_tensor_message_to_numpy,
        StringTensor: convert_tensor_message_to_numpy,
        DiscreteSpace: convert_discrete_space_message,
        NamedDiscreteSpace: convert_named_discrete_space_message,
        CommandlineSpace: convert_commandline_space_message,
        BooleanRange: convert_range_message,
        Int64Range: convert_range_message,
        FloatRange: convert_range_message,
        DoubleRange: convert_range_message,
        StringSpace: convert_string_space,
        BooleanSequenceSpace: convert_sequence_space,
        ByteSequenceSpace: convert_sequence_space,
        BytesSequenceSpace: convert_sequence_space,
        Int64SequenceSpace: convert_sequence_space,
        FloatSequenceSpace: convert_sequence_space,
        DoubleSequenceSpace: convert_sequence_space,
        StringSequenceSpace: convert_sequence_space,
        BooleanBox: convert_box_message,
        ByteBox: convert_box_message,
        Int64Box: convert_box_message,
        FloatBox: convert_box_message,
        DoubleBox: convert_box_message,
    }

    res = TypeBasedConverter(conversion_map)
    conversion_map[Event] = EventMessageConverter(res)
    conversion_map[ListEvent] = ListEventMessageConverter(conversion_map[Event])
    conversion_map[DictEvent] = DictEventMessageConverter(conversion_map[Event])

    conversion_map[Space] = SpaceMessageConverter(res)
    conversion_map[ListSpace] = ListSpaceMessageConverter(conversion_map[Space])
    conversion_map[DictSpace] = DictSpaceMessageConverter(conversion_map[Space])
    conversion_map[ActionSpace] = ActionSpaceMessageConverter(res)
    conversion_map[ObservationSpace] = ObservationSpaceMessageConverter(res)

    conversion_map[any_pb2.Any] = ProtobufAnyConverter(
        unpacker=ProtobufAnyUnpacker(), message_converter=res
    )

    conversion_map[Opaque] = make_opaque_message_default_converter()
    return res


def to_event_message_default_converter() -> ToEventMessageConverter:
    conversion_map = {
        bool: convert_trivial,
        int: convert_trivial,
        float: convert_trivial,
        str: convert_trivial,
        np.ndarray: NumpyToTensorMessageConverter(),
    }
    type_based_converter = TypeBasedConverter(conversion_map)
    res = ToEventMessageConverter(type_based_converter)
    conversion_map[list] = ToListEventMessageConverter(res)
    conversion_map[dict] = ToDictEventMessageConverter(res)
    return res


range_type_default_min_map: DictType[Type, Any] = {
    BooleanRange: False,
    Int64Range: np.iinfo(np.int64).min,
    FloatRange: np.float32(np.NINF),
    DoubleRange: np.float64(np.NINF),
}

range_type_default_max_map: DictType[Type, Any] = {
    BooleanRange: True,
    Int64Range: np.iinfo(np.int64).max,
    FloatRange: np.float32(np.PINF),
    DoubleRange: np.float64(np.PINF),
}


def convert_range_message(
    range: Union[BooleanRange, Int64Range, FloatRange, DoubleRange]
) -> Scalar:
    range_type = type(range)
    min = range.min if range.HasField("min") else range_type_default_min_map[range_type]
    max = range.max if range.HasField("max") else range_type_default_max_map[range_type]
    return Scalar(
        name=None, min=min, max=max, dtype=proto_type_to_dtype_map[range_type]
    )


class ToRangeMessageConverter:
    dtype_to_type_map: DictType[Type, Type]

    def __init__(self):
        self.dtype_to_type_map = {
            np.bool_: BooleanRange,
            np.int8: Int64Range,
            np.int64: Int64Range,
            np.float32: FloatRange,
            np.float64: DoubleRange,
        }

    def __call__(
        self, scalar: Scalar
    ) -> Union[BooleanRange, Int64Range, FloatRange, DoubleRange]:
        return self.dtype_to_type_map[np.dtype(scalar.dtype).type](
            min=scalar.min, max=scalar.max
        )


convert_to_range_message = ToRangeMessageConverter()


def convert_box_message(
    box: Union[BooleanBox, ByteBox, Int64Box, FloatBox, DoubleBox]
) -> Box:
    return Box(
        low=convert_tensor_message_to_numpy(box.low),
        high=convert_tensor_message_to_numpy(box.high),
        name=None,
        dtype=proto_type_to_dtype_map[type(box)],
    )


class ToBoxMessageConverter:
    dtype_to_type_map: DictType[Type, Type]

    def __init__(self):
        self.dtype_to_type_map = {
            np.bool_: BooleanBox,
            np.int8: ByteBox,
            np.int64: Int64Box,
            np.float32: FloatBox,
            np.float64: DoubleBox,
        }

    def __call__(
        self, box: Box
    ) -> Union[BooleanBox, ByteBox, Int64Box, FloatBox, DoubleBox]:
        return self.dtype_to_type_map[np.dtype(box.dtype).type](
            low=convert_numpy_to_tensor_message(box.low),
            high=convert_numpy_to_tensor_message(box.high),
        )


convert_to_box_message = ToBoxMessageConverter()


def convert_discrete_space_message(message: DiscreteSpace) -> Discrete:
    return Discrete(n=message.n, name=None)


def convert_to_discrete_space_message(space: Discrete) -> DiscreteSpace:
    return DiscreteSpace(n=space.n)


def convert_named_discrete_space_message(message: NamedDiscreteSpace) -> NamedDiscrete:
    return NamedDiscrete(items=message.name, name=None)


def convert_commandline_space_message(message: CommandlineSpace) -> Commandline:
    return Commandline(
        items=[
            CommandlineFlag(name=name, flag=name, description="")
            for name in message.name
        ],
        name=None,
    )


def convert_to_named_discrete_space_message(space: NamedDiscrete) -> NamedDiscreteSpace:
    return NamedDiscreteSpace(name=space.names)


def convert_sequence_space(
    seq: Union[
        BooleanSequenceSpace,
        Int64SequenceSpace,
        FloatSequenceSpace,
        DoubleSequenceSpace,
        BytesSequenceSpace,
        StringSequenceSpace,
    ]
) -> Sequence:
    scalar_range = (
        convert_range_message(seq.scalar_range)
        if hasattr(seq, "scalar_range")
        else None
    )
    length_range = convert_range_message(seq.length_range)
    return Sequence(
        name=None,
        size_range=(length_range.min, length_range.max),
        dtype=proto_type_to_dtype_map[type(seq)],
        scalar_range=scalar_range,
    )


class ToRangedSequenceMessageConverter:
    dtype_to_type_map: DictType[Type, Type]

    def __init__(self):
        self.dtype_to_type_map = {
            np.bool_: BooleanSequenceSpace,
            np.int8: ByteSequenceSpace,
            np.int64: Int64SequenceSpace,
            np.float32: FloatSequenceSpace,
            np.float64: DoubleSequenceSpace,
        }

    def __call__(
        self, seq: Sequence
    ) -> Union[
        BooleanSequenceSpace,
        Int64SequenceSpace,
        FloatSequenceSpace,
        DoubleSequenceSpace,
    ]:
        return self.dtype_to_type_map[np.dtype(seq.dtype).type](
            length_range=Int64Range(min=seq.size_range[0], max=seq.size_range[1]),
            scalar_range=convert_to_range_message(seq.scalar_range),
        )


convert_to_ranged_sequence_space = ToRangedSequenceMessageConverter()


def convert_to_string_sequence_space(seq: Sequence) -> StringSequenceSpace:
    return StringSpace(
        length_range=Int64Range(min=seq.size_range[0], max=seq.size_range[1])
    )


def convert_to_bytes_sequence_space(seq: Sequence) -> BytesSequenceSpace:
    return BytesSequenceSpace(
        length_range=Int64Range(min=seq.size_range[0], max=seq.size_range[1])
    )


def convert_string_space(s: StringSpace) -> Sequence:
    return convert_sequence_space(s)


def convert_to_string_space(s: Sequence) -> StringSpace:
    return StringSpace(
        length_range=Int64Range(min=s.size_range[0], max=s.size_range[1])
    )


class ToSequenceSpaceMessageConverter:
    dtype_map: DictType[
        Type,
        Callable[
            [Sequence],
            Union[
                BooleanSequenceSpace,
                BytesSequenceSpace,
                Int64SequenceSpace,
                FloatSequenceSpace,
                DoubleSequenceSpace,
                StringSequenceSpace,
            ],
        ],
    ]

    def __init__(self):
        self.dtype_map = {
            bool: convert_to_ranged_sequence_space,
            np.bool_: convert_to_ranged_sequence_space,
            np.int8: convert_to_bytes_sequence_space,
            np.int64: convert_to_ranged_sequence_space,
            int: convert_to_ranged_sequence_space,
            np.float32: convert_to_ranged_sequence_space,
            np.float64: convert_to_ranged_sequence_space,
            float: convert_to_ranged_sequence_space,
            str: convert_to_string_space,
        }

    def __call__(
        self, seq: Sequence
    ) -> Union[
        BooleanSequenceSpace,
        BytesSequenceSpace,
        Int64SequenceSpace,
        FloatSequenceSpace,
        DoubleSequenceSpace,
        StringSequenceSpace,
    ]:
        return self.dtype_map[seq.dtype](seq)


convert_to_sequence_space_message = ToSequenceSpaceMessageConverter()


class SpaceMessageConverter:
    message_converter: TypeBasedConverter

    def __init__(self, message_converter: TypeBasedConverter):
        self.message_converter = message_converter

    def __call__(
        self, space: Space
    ) -> Union[Dict, Discrete, NamedDiscrete, Scalar, Tuple, Box, Sequence]:
        field = space.WhichOneof("value")
        if field is None:
            return None

        res = self.message_converter(getattr(space, field))
        return res


class ToSpaceMessageConverter:
    converter: TypeBasedConverter
    type_field_map: DictType[Type, str]

    def __init__(self, converter: TypeBasedConverter):
        self.converter = converter
        self.type_field_map = {
            ListSpace: "space_list",
            DictSpace: "space_dict",
            DiscreteSpace: "discrete",
            NamedDiscreteSpace: "named_discrete",
            BooleanRange: "boolean_value",
            Int64Range: "int64_value",
            FloatRange: "float_value",
            DoubleRange: "double_value",
            StringSpace: "string_value",
            BooleanSequenceSpace: "boolean_sequence",
            BytesSequenceSpace: "bytes_sequence",
            ByteSequenceSpace: "byte_sequence",
            Int64SequenceSpace: "int64_sequence",
            FloatSequenceSpace: "float_sequence",
            DoubleSequenceSpace: "double_sequence",
            StringSequenceSpace: "string_sequence",
            BooleanBox: "boolean_box",
            ByteBox: "byte_box",
            Int64Box: "int64_box",
            FloatBox: "float_box",
            DoubleBox: "double_box",
            any_pb2.Any: "any_value",
        }

    def __call__(
        self, space: Union[Tuple, Dict, Discrete, NamedDiscrete, Sequence, Box, Scalar]
    ) -> Space:
        converted_space = self.converter(space)
        res = Space()
        if isinstance(converted_space, Message):
            getattr(res, self.type_field_map[type(converted_space)]).CopyFrom(
                converted_space
            )
        else:
            setattr(res, self.type_field_map[type(converted_space)], converted_space)
        return res


class ListSpaceMessageConverter:
    space_message_converter: SpaceMessageConverter

    def __init__(self, space_message_converter: SpaceMessageConverter):
        self.space_message_converter = space_message_converter

    def __call__(self, list_space: ListSpace) -> Tuple:
        return Tuple(
            spaces=[self.space_message_converter(space) for space in list_space.space],
            name=None,
        )


class ToListSpaceMessageConverter:
    to_space_converter: ToSpaceMessageConverter

    def __init__(self, to_space_converter: ToSpaceMessageConverter):
        self.to_space_converter = to_space_converter

    def __call__(self, spaces: Tuple) -> ListSpace:
        return ListSpace(
            space=[self.to_space_converter(space) for space in spaces.spaces]
        )


class DictSpaceMessageConverter:
    space_message_converter: SpaceMessageConverter

    def __init__(self, space_message_converter: SpaceMessageConverter):
        self.space_message_converter = space_message_converter

    def __call__(self, dict_space: DictSpace) -> Dict:
        return Dict(
            spaces={
                key: self.space_message_converter(space)
                for key, space in dict_space.space.items()
            },
            name=None,
        )


class ToDictSpaceMessageConverter:
    to_space_converter: ToSpaceMessageConverter

    def __init__(self, to_space_converter: ToSpaceMessageConverter):
        self.to_space_converter = to_space_converter

    def __call__(self, d: Dict) -> DictSpace:
        return DictSpace(
            space={key: self.to_space_converter(val) for key, val in d.spaces.items()}
        )


def to_space_message_default_converter() -> ToSpaceMessageConverter:
    conversion_map = {
        Discrete: convert_to_discrete_space_message,
        NamedDiscrete: convert_to_named_discrete_space_message,
        Scalar: convert_to_range_message,
        Sequence: convert_to_sequence_space_message,
        Box: convert_to_box_message,
    }
    type_based_converter = TypeBasedConverter(conversion_map)
    res = ToSpaceMessageConverter(type_based_converter)
    conversion_map[Tuple] = ToListSpaceMessageConverter(res)
    conversion_map[Dict] = ToDictSpaceMessageConverter(res)
    return res


class OpaqueMessageConverter:
    """Converts <compiler_gym.service.proto.Opaque> message based on its format descriptor."""

    format_coverter_map: DictType[str, Callable[[bytes], Any]]

    def __init__(self, format_coverter_map=None):
        self.format_coverter_map = (
            {} if format_coverter_map is None else format_coverter_map
        )

    def __call__(self, message: Opaque) -> Any:
        return self.format_coverter_map[message.format](message.data)


def make_opaque_message_default_converter():
    return OpaqueMessageConverter(
        {"json://networkx/MultiDiGraph": _json2nx, "json://": bytes_to_json}
    )


def bytes_to_json(data: bytes):
    return json.loads(data.decode("utf-8"))


def _json2nx(data: bytes):
    json_data = json.loads(data.decode("utf-8"))
    return nx.readwrite.json_graph.node_link_graph(
        json_data, multigraph=True, directed=True
    )


message_default_converter: TypeBasedConverter = make_message_default_converter()
