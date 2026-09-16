Python API
==========

The application programming interface (API) below is the set of Python objects
KCoral exposes to callers. Install the client for program construction and
submission; install the ``server`` extra when using ``create_app``.

.. currentmodule:: kcoral

Client
------

.. autoclass:: Client
   :members: function, execute, health, target, close
   :special-members: __init__, __enter__, __exit__

Remote functions
----------------

.. autoclass:: RemoteFunction
   :members: remote, execute, build_program
   :special-members: __call__

Program construction
--------------------

.. autoclass:: Program
   :members: instructions, upload, upload_file, upload_folder, get_function, run, return_, return_file, return_folder

.. autoclass:: Register
   :members: id

Results
-------

.. autoclass:: ProgramResult
   :members: completed
   :special-members: __getitem__

.. autoclass:: ReturnedFile
   :members: read_bytes, save

.. autoclass:: ReturnedFolder
   :members: files, directories, save

.. _python-errors:

Errors
------

``Client.execute`` and ``RemoteFunction.execute`` represent instruction failures
as a :class:`ProgramResult` with status ``FAILED``. ``RemoteFunction.remote``
instead raises :class:`RemoteExecutionError` retaining that outcome.
The other exceptions below describe failures to obtain a valid program outcome.
See :doc:`../client-guide/protocol` for server error codes.

.. autoclass:: RemoteExecutionError

.. autoclass:: KCoralError

.. autoclass:: TransportError

.. autoclass:: ProtocolError

GPU utilities
-------------

Uploaded Python can import these functions from ``kcoral.builtins``.

.. autofunction:: kcoral.builtins.compile_tirx

.. autofunction:: kcoral.builtins.benchmark

.. _server-integration:

Server integration
------------------

.. autoclass:: ServerConfig

The :doc:`../server-guide/launch-the-server` page explains each field, the corresponding
command-line option and differences between command-line and Python defaults.

.. autofunction:: create_app

The returned `FastAPI application <https://fastapi.tiangolo.com/reference/fastapi/>`_
can be served with uvicorn or another compatible HTTP server.

.. autofunction:: parse_program

.. py:class:: kcoral.schemas.Program

   Parsed protocol data returned by :func:`kcoral.parse_program`. Its
   ``instructions`` contain validated protocol instruction objects, and its
   ``options`` contain the parsed request options. This is different from
   :class:`kcoral.Program`, the client-side request builder.

   This object is intended for server integration. Client applications should
   normally construct a :class:`kcoral.Program` and call :meth:`kcoral.Client.execute`.

.. py:exception:: kcoral.errors.ValidationError

   Raised when protocol data violates the schema, including malformed fields,
   duplicate identifiers, invalid paths or references to later instructions.

   The HTTP front-end turns this exception into a request validation response.
