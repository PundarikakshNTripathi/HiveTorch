from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class JoinRequest(_message.Message):
    __slots__ = ("client_id", "client_index")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    CLIENT_INDEX_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    client_index: int
    def __init__(self, client_id: _Optional[str] = ..., client_index: _Optional[int] = ...) -> None: ...

class JoinResponse(_message.Message):
    __slots__ = ("accepted", "session_id", "num_rounds", "message")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_ROUNDS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    session_id: str
    num_rounds: int
    message: str
    def __init__(self, accepted: _Optional[bool] = ..., session_id: _Optional[str] = ..., num_rounds: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class ModelRequest(_message.Message):
    __slots__ = ("client_id", "session_id", "round")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ROUND_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    session_id: str
    round: int
    def __init__(self, client_id: _Optional[str] = ..., session_id: _Optional[str] = ..., round: _Optional[int] = ...) -> None: ...

class ModelResponse(_message.Message):
    __slots__ = ("current_round", "participate", "finished", "global_weights")
    CURRENT_ROUND_FIELD_NUMBER: _ClassVar[int]
    PARTICIPATE_FIELD_NUMBER: _ClassVar[int]
    FINISHED_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    current_round: int
    participate: bool
    finished: bool
    global_weights: bytes
    def __init__(self, current_round: _Optional[int] = ..., participate: _Optional[bool] = ..., finished: _Optional[bool] = ..., global_weights: _Optional[bytes] = ...) -> None: ...

class ClientUpdate(_message.Message):
    __slots__ = ("client_id", "session_id", "round", "weights", "num_samples", "loss")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ROUND_FIELD_NUMBER: _ClassVar[int]
    WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    NUM_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    LOSS_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    session_id: str
    round: int
    weights: bytes
    num_samples: int
    loss: float
    def __init__(self, client_id: _Optional[str] = ..., session_id: _Optional[str] = ..., round: _Optional[int] = ..., weights: _Optional[bytes] = ..., num_samples: _Optional[int] = ..., loss: _Optional[float] = ...) -> None: ...

class ServerAck(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: _Optional[bool] = ..., message: _Optional[str] = ...) -> None: ...

class StatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("current_round", "num_rounds", "finished", "round_accuracies")
    CURRENT_ROUND_FIELD_NUMBER: _ClassVar[int]
    NUM_ROUNDS_FIELD_NUMBER: _ClassVar[int]
    FINISHED_FIELD_NUMBER: _ClassVar[int]
    ROUND_ACCURACIES_FIELD_NUMBER: _ClassVar[int]
    current_round: int
    num_rounds: int
    finished: bool
    round_accuracies: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, current_round: _Optional[int] = ..., num_rounds: _Optional[int] = ..., finished: _Optional[bool] = ..., round_accuracies: _Optional[_Iterable[float]] = ...) -> None: ...
