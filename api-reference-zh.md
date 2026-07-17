# GPU Server v2 API 参考

## 1. 概述

GPU Server v2 是一个同步远程 GPU 执行服务。GPU 指图形处理器，API 指应用程序编程接口。

客户端在一次请求中上传：

- 一个入口脚本，默认路径为 `main.py`；
- 零个或多个脚本运行所需的工件文件；
- 可选的执行设置，包括语言、入口位置和超时时间。

这里的“工件”指配置文件、输入数据、Python 模块、动态链接库以及其他供入口脚本读取的文件。

服务端为请求创建独立工作目录，加载入口文件，调用其中无参数的入口函数，然后把函数返回值传回调用方。入口文件和入口函数默认为 `main.py` 中的 `main()`。

协议使用 `language` 字段描述入口脚本的语言。第一版只支持 Python，协议形状为后续语言预留。

协议中没有指令列表、操作码、寄存器、远程函数名和输出文件声明。模块加载、张量构造、GPU 内核执行、正确性检查和性能测量都由上传的脚本完成。

第一版只面向可信客户端。上传的代码可以使用工作进程权限执行任意操作，该接口不提供安全沙箱。

---

## 2. 接口

### 2.1 `POST /execute`

上传并在一张 GPU 上执行一个程序。

该请求采用同步模式。请求在排队和执行期间保持连接，成功响应携带入口函数的返回值。

#### 2.1.1 请求格式

内容类型：

```text
multipart/form-data
```

`multipart/form-data` 是一种 HTTP 多段请求格式，可以在一个请求体中携带 JSON 元数据和多个二进制文件。HTTP 指超文本传输协议，JSON 指 JavaScript Object Notation，一种结构化文本数据格式。

请求体支持以下部分：

| 部分名称 | 内容类型 | 必需 | 说明 |
|---|---|---:|---|
| `job` | `application/json` | 是 | 执行设置和文件 manifest |
| `blob:<sha256>` | `application/octet-stream` | 是 | 由完整 SHA-256 哈希标识的文件内容 |

请求必须包含且只包含一个 `job` part，并为 `job.files` 引用的每个唯一 hash 包含一个 blob part。

manifest 指描述文件路径及其内容引用的 JSON 数据。`job.files` 把工作目录相对路径映射到 blob。上传内容不在 multipart 名称中携带文件路径。

blob 部分名称示例：

```text
blob:81d4<其余 SHA-256 十六进制字符>
blob:97ab<其余 SHA-256 十六进制字符>
```

每个 `blob` 值必须是 64 个小写十六进制字符。multipart part 名由 `blob:` 和该值拼接而成。服务端重新计算每个 part 的 SHA-256 并验证名称。SHA-256 是一种密码学哈希函数，这里同时用于内容寻址、完整性校验和文件缓存。

请求必须携带 `job.files` 引用的每个唯一 blob，即使服务端缓存中已经存在相同内容。一个 blob 可以被多个路径引用，multipart 中只上传一次。缺少引用的 blob、hash 不一致、重复 blob part 或未被 manifest 引用的 blob 都返回 `invalid_request`。part 顺序不影响请求语义。

##### `job` 对象

必需的 `job` 部分格式如下：

```json
{
  "language": "python",
  "entry": {
    "file": "main.py",
    "function": "main"
  },
  "files": {
    "main.py": {
      "blob": "<main_sha256>"
    },
    "data/input.bin": {
      "blob": "<input_sha256>"
    }
  },
  "timeout_seconds": 60,
  "stdout_limit_bytes": 1048576,
  "stderr_limit_bytes": 1048576
}
```

| 字段 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `language` | 否 | `"python"` | 入口脚本的语言 |
| `entry` | 否 | 见下 | 入口位置 |
| `entry.file` | 否 | `"main.py"` | 入口文件在工作目录中的路径 |
| `entry.function` | 否 | `"main"` | 入口函数名 |
| `files` | 是 | 无 | 从工作目录相对路径到文件内容引用的映射 |
| `files.<path>.blob` | 是 | 无 | 对应文件内容的完整 SHA-256 哈希 |
| `timeout_seconds` | 否 | 服务端配置 | 正数，表示执行超时时间，单位为秒 |
| `stdout_limit_bytes` | 否 | 服务端配置 | 响应中标准输出的最大字节数 |
| `stderr_limit_bytes` | 否 | 服务端配置 | 响应中标准错误的最大字节数 |

`files` 是唯一必需字段。省略其他字段时，服务端使用默认语言和入口，相当于执行：

```python
from main import main

return_value = main()
```

`language` 第一版只接受 `"python"`。其他取值返回 `invalid_request`。该字段决定服务端如何加载入口文件和调用入口函数；后续语言必须在协议修订中定义各自的加载和调用规则。

`entry.file` 必须满足 2.1.1 节的文件路径规则，并且必须是 `files` 中的一个 key。`entry.function` 必须是入口文件中定义的无参数可调用函数。

服务端拒绝未知字段，避免客户端误以为某项未支持的设置已经生效。

`job` 不声明输出。应用层结果只有入口函数的返回值。

`stdout_limit_bytes` 和 `stderr_limit_bytes` 必须是非负整数，并且不能超过服务端配置的上限。`0` 表示响应不携带对应输出。限制只作用于 API 响应和 SDK 执行结果；第 5 章定义的日志始终保存完整输出。

文档其余部分使用默认入口 `main.py` 和 `main()` 描述行为；除非特别说明，这些描述对自定义 `entry` 同样成立，把 `main.py` 替换为 `entry.file`、`main()` 替换为 `entry.function` 即可。

##### 文件路径规则

`files` 中的所有 key 都必须满足以下条件：

- 使用 `/` 作为路径分隔符；
- 使用相对于工作目录的路径；
- 不包含空路径段、`.` 或 `..`；
- 不包含空字符；
- 不以 `/` 开头；
- 不允许通过符号链接指向工作目录之外；
- 在一次请求中保持唯一；
- 不覆盖服务端创建的内部文件。

路径规范化之后必须与客户端提交的路径完全相同：

| `files` key | 处理结果 |
|---|---|
| `data/input.bin` | 接受 |
| `./input.bin` | 拒绝 |
| `data/../input.bin` | 拒绝 |
| `/etc/passwd` | 拒绝 |

服务端拒绝重复 JSON key 和路径规范化后发生的冲突，并按需创建父目录。

##### Python 入口约定

`language` 为 `"python"` 时，入口文件必须定义一个与 `entry.function` 同名的可调用函数。使用默认值时即：

```python
def main():
    ...
    return result
```

入口函数必须满足：

- 不接收参数；
- 可以导入同一请求上传的其他 Python 文件；
- 可以使用相对于当前工作目录的路径读取工件；
- 运行时只能看到一张 GPU，该设备显示为 `cuda:0`；
- 返回一个受支持的值；
- 在请求超时前完成。

服务端在导入入口文件之前，把当前工作目录切换到请求工作目录，并在请求期间将该目录放到 Python 模块搜索路径的最前面。

服务端以模块方式导入入口文件。文件顶层代码会在调用入口函数之前执行。建议把实际工作放在入口函数内，便于确定失败位置和测量执行时间。

服务端不传递命令行参数，不注入应用对象，也不查找 `entry.function` 以外的函数名。

##### 脚本执行环境

每张 GPU 对应一个工作槽位，同一槽位上的请求串行执行。

服务端在执行脚本前限制 GPU 可见范围，使分配到的物理 GPU 在脚本中显示为：

```text
cuda:0
```

脚本使用服务端预先配置的 Python 解释器和已安装依赖。客户端不能通过该接口选择其他解释器或安装依赖。

服务端提供以下环境变量：

| 环境变量 | 说明 |
|---|---|
| `GPU_SERVER_REQUEST_ID` | 用于日志和问题定位的唯一请求标识 |
| `GPU_SERVER_WORK_DIR` | 本次请求工作目录的绝对路径 |

程序应优先使用相对路径访问上传的工件。工作目录绝对路径仅对当前请求有效，不应持久化使用。

#### 2.1.2 响应

##### 请求标识

服务端在 `/execute` 请求进入处理函数后、解析请求体之前生成一个 UUID v4。UUID 指通用唯一标识符，v4 表示该标识符使用随机数生成。

请求标识使用小写标准 UUID 字符串：

```text
7f61b94e-034a-4e80-b67d-eca52bb952cc
```

该值在协议中统一命名为 `request_id`，并贯穿请求解析、排队、工作进程执行、日志记录和响应生成。

每个 `/execute` 响应都通过 HTTP 响应头返回请求标识：

```http
X-Request-ID: 7f61b94e-034a-4e80-b67d-eca52bb952cc
```

成功和错误响应的 JSON 元数据也包含相同的 `request_id`。对于字节串和张量返回值，该字段位于 multipart 响应的 `result` 部分。

客户端不能指定或覆盖 `request_id`。同一次 HTTP 请求在服务端内部始终使用同一个值；客户端重试会产生新的 `request_id`。如果以后需要识别业务层重复提交，应增加独立的幂等键，不能使用 `request_id` 表示幂等关系。

`GPU_SERVER_REQUEST_ID` 环境变量的值与响应中的 `request_id` 完全一致。

##### 返回值

入口函数可以返回由以下类型递归组成的值：

- `None`、`bool`、`int`、有限的 `float` 和 `str`；
- `bytes`、`bytearray` 和 `memoryview`；
- 实现 DLPack 生产者协议的张量对象，包括 `tvm_ffi.Tensor`；
- `list`；
- `tuple`；
- 键为字符串的 `dict`。

列表、元组和字典可以在任意层级包含上述类型。一个返回值可以同时包含多个字节串和多个张量。

DLPack 是用于内存张量交换的标准协议。DLPack 生产者对象必须提供 `__dlpack__()` 和 `__dlpack_device__()`。入口函数可以直接返回的常见张量类型包括：

- `torch.Tensor`；
- `numpy.ndarray`；
- `cupy.ndarray`；
- `jax.Array`；
- `tvm_ffi.Tensor`；
- 其他实现 `__dlpack__()` 和 `__dlpack_device__()` 的对象。

服务端先把张量归一化为 `apache-tvm-ffi` 包中的 `tvm_ffi.Tensor`。已有的 `tvm_ffi.Tensor` 直接使用；其他 DLPack 生产者通过 `tvm_ffi.from_dlpack(..., require_contiguous=True)` 转换：

```python
import tvm_ffi


def normalize_tensor(value):
    if isinstance(value, tvm_ffi.Tensor):
        tensor = value
    elif hasattr(value, "__dlpack__") and hasattr(value, "__dlpack_device__"):
        tensor = tvm_ffi.from_dlpack(value, require_contiguous=True)
    else:
        raise UnsupportedReturnTypeError()

    if not tensor.is_contiguous():
        raise ReturnSerializationError("tensor must be C-contiguous")

    return tensor
```

第一版只接受 C 连续的稠密张量。DLPack 生产者导出的张量不连续，或无法由 `tvm_ffi.from_dlpack` 导入时，服务端返回 `invalid_return_value`。服务端不尝试自动连续化。未实现 DLPack 生产者协议的其他对象同样返回 `invalid_return_value`。

参见官方 [`tvm_ffi.from_dlpack` 文档](https://tvm.apache.org/ffi/reference/python/generated/tvm_ffi.from_dlpack.html)。

###### 返回值描述树

服务端将 Python 返回值递归编码为一棵描述树：

| Python 值 | 描述节点 |
|---|---|
| 完全可表示为 JSON 的子树 | `{"type": "json", "value": ...}` |
| `bytes`、`bytearray`、`memoryview` | `{"type": "bytes", ...}` |
| 实现 DLPack 生产者协议的 C 连续张量 | `{"type": "tensor", ...}` |
| `list` | `{"type": "list", "items": [...]}` |
| `tuple` | `{"type": "tuple", "items": [...]}` |
| `dict` | `{"type": "dict", "items": {...}}` |

JSON 节点可以包含：

- `null`；
- 布尔值；
- 整数；
- 有限浮点数；
- 字符串；
- 只包含 JSON 值的数组；
- 键为字符串、值为 JSON 值的对象。

元组在描述树中保留为 `tuple`，客户端可以据此恢复元组。正无穷、负无穷和非数值等非有限浮点数不属于 JSON 值。

如果一个完整子树都可以表示为 JSON，服务端将该子树合并为一个 `json` 节点，避免为每个标量生成描述节点。

###### 纯 JSON 返回值

返回值完全由 JSON 值组成时，响应内容类型为 `application/json`。

`main.py` 示例：

```python
def main():
    return {
        "correct": True,
        "median_ms": 0.128,
        "samples_ms": [0.127, 0.128, 0.131],
    }
```

成功响应：

```json
{
  "status": "ok",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "return": {
    "type": "json",
    "value": {
      "correct": true,
      "median_ms": 0.128,
      "samples_ms": [0.127, 0.128, 0.131]
    }
  },
  "elapsed_ms": 1245.6,
  "queue_ms": 18.2,
  "stdout": "",
  "stderr": ""
}
```

###### 包含二进制值的返回值

返回值树中出现字节串或张量时，响应内容类型为 `multipart/form-data`，包含：

| 部分名称 | 内容类型 | 说明 |
|---|---|---|
| `result` | `application/json` | 执行元数据和完整返回值描述树 |
| `return:<index>` | `application/octet-stream` | 一个字节串或张量的原始字节 |

`<index>` 从 `0` 开始。服务端按深度优先遍历顺序为二进制值分配 part 标识。part 标识在 JSON 中使用字符串表示，例如：

```json
"part": "return:0"
```

该字符串与 multipart 部分的 `name` 完全相同：

```http
Content-Disposition: form-data; name="return:0"
Content-Type: application/octet-stream
```

客户端必须使用描述节点中的 `part` 字段查找数据，不应自行推算编号或解析标识符中的数字。part 标识只在当前 HTTP 响应中有效。

每个二进制节点同时携带：

| 字段 | 说明 |
|---|---|
| `part` | 当前响应中的 multipart 部分名称 |
| `size` | 原始数据字节数 |
| `sha256` | 原始数据的完整 SHA-256 哈希 |

`part` 用于定位数据，`sha256` 用于完整性校验。多个返回值节点可以引用同一个 part，以复用内容完全相同的二进制数据。

###### 字节串节点

字节串节点格式：

```json
{
  "type": "bytes",
  "part": "return:0",
  "size": 15,
  "sha256": "<sha256>"
}
```

对应的 multipart 部分保存 `bytes`、`bytearray` 或 `memoryview` 的原始字节。

###### 张量节点

张量节点格式：

```json
{
  "type": "tensor",
  "dtype": "float16",
  "shape": [32, 128],
  "part": "return:0",
  "size": 8192,
  "sha256": "<sha256>"
}
```

服务端将 DLPack 生产者归一化为 `tvm_ffi.Tensor`，验证其采用 C 连续布局，执行设备同步，然后将其复制到主机内存。张量字节使用连续的行优先顺序存储，多字节标量使用小端字节序。非连续张量返回 `invalid_return_value`。

张量节点中的：

- `dtype` 表示元素数据类型；
- `shape` 表示各维长度；
- `size` 必须等于形状中各维长度的乘积乘以单个元素的字节数。

###### 嵌套返回值示例

当 `output_tensor` 是任意受支持的 DLPack 生产者对象时，`main()` 可以返回：

```python
def main():
    return {
        "correct": True,
        "metrics": {
            "median_ms": 0.128,
            "samples_ms": [0.127, 0.128, 0.131],
        },
        "outputs": [
            output_tensor,
            b"binary metadata",
        ],
    }
```

`result` 部分中的返回值描述树为：

```json
{
  "status": "ok",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "return": {
    "type": "dict",
    "items": {
      "correct": {
        "type": "json",
        "value": true
      },
      "metrics": {
        "type": "json",
        "value": {
          "median_ms": 0.128,
          "samples_ms": [0.127, 0.128, 0.131]
        }
      },
      "outputs": {
        "type": "list",
        "items": [
          {
            "type": "tensor",
            "dtype": "float16",
            "shape": [32, 128],
            "part": "return:0",
            "size": 8192,
            "sha256": "<sha256>"
          },
          {
            "type": "bytes",
            "part": "return:1",
            "size": 15,
            "sha256": "<sha256>"
          }
        ]
      }
    }
  },
  "elapsed_ms": 1245.6,
  "queue_ms": 18.2,
  "stdout": "",
  "stderr": ""
}
```

multipart 响应还包含 `return:0` 和 `return:1` 两个二进制部分。

###### 类型和序列化限制

返回值不属于本节列出的类型时，服务端返回 `invalid_return_value`。例如：

- 生成器和迭代器；
- 打开的文件对象；
- 未实现 DLPack 生产者协议的其他 Python 类实例；
- 函数和模块；
- 键不是字符串的字典。

服务端应在错误信息中提供无法编码的值路径，例如：

```json
{
  "status": "error",
  "error": "invalid_return_value",
  "message": "unsupported value at $.outputs[2].metadata",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc"
}
```

服务端不保留 Python 对象引用关系。同一个对象在返回值树中出现多次时，客户端得到多个独立引用；对应二进制内容可以共享同一个 part。

循环引用无法表示为有限描述树。服务端在递归构建描述树时必须主动检查循环引用，不能等待 Python 达到递归深度限制。

检查时记录当前递归路径上 `list`、`tuple` 和 `dict` 的对象标识：

1. 进入容器前，如果它的对象标识已经位于当前递归路径中，则发现循环引用；
2. 进入容器时，将对象标识加入当前递归路径；
3. 完成该容器编码后，将对象标识移出当前递归路径。

该集合只记录当前递归路径，不记录所有已经访问的对象。因此，多个位置可以引用同一个非循环对象；这些位置会分别编码。

发现循环引用时，服务端返回 `invalid_return_value`，并在错误信息中携带发现循环的位置：

```json
{
  "status": "error",
  "error": "invalid_return_value",
  "message": "circular reference at $.outputs[1]",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc"
}
```

描述树构建完成后，服务端使用 Python `json.dumps` 生成 `result`。调用时保持默认的 `check_circular=True`，并设置 `allow_nan=False`，对循环引用和非有限浮点数再做一次校验。自定义递归检查仍然必需，因为循环可能在描述树构建完成之前发生。

服务端必须配置：

- 最大嵌套深度；
- 最大描述节点数量；
- 最大 JSON 元数据字节数；
- 单个二进制值最大字节数；
- 整个响应最大字节数。

服务端在发送 HTTP 响应头之前完成返回值遍历，并将所有二进制内容序列化到临时文件。这样可以在遍历、同步或序列化失败时返回完整的 JSON 错误响应。

##### 成功响应元数据

每个成功响应都包含：

| 字段 | 说明 |
|---|---|
| `status` | 固定为 `"ok"` |
| `request_id` | 服务端为本次 HTTP 请求生成的 UUID v4 |
| `return` | 入口函数返回值的类型和表示 |
| `elapsed_ms` | 从开始导入入口文件到返回值序列化完成的时间 |
| `queue_ms` | 等待可用 GPU 工作进程的时间 |
| `stdout` | 导入入口文件和执行入口函数期间捕获的标准输出 |
| `stderr` | 导入入口文件和执行入口函数期间捕获的标准错误 |

`elapsed_ms` 包含：

- 导入入口文件；
- 执行文件顶层代码；
- 执行入口函数；
- 同步并序列化返回值。

`elapsed_ms` 不包含：

- HTTP 请求上传时间；
- 排队时间；
- HTTP 响应下载时间。

GPU 内核性能应由入口函数使用适合该运行环境的 GPU 同步和计时方式测量。`elapsed_ms` 描述服务端执行总耗时，不能作为内核性能结果。

服务端使用 `stdout_limit_bytes` 和 `stderr_limit_bytes` 限制响应中的标准输出和标准错误。限制按原始字节数计算，响应保留每个流的前 N 个字节，并使用 UTF-8 解码；无效字节序列使用替换字符。发生截断时，响应额外包含：

```json
{
  "stdout_truncated": true,
  "stderr_truncated": false
}
```

#### 2.1.3 错误

所有错误响应都使用 `application/json`。

通用格式：

```json
{
  "status": "error",
  "error": "<error-type>",
  "message": "<human-readable message>",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "stdout": "",
  "stderr": ""
}
```

错误响应中的 `request_id` 是小写标准 UUID v4，与 `X-Request-ID` 响应头完全一致。如果脚本已经开始执行，响应会包含 `stdout` 和 `stderr`。

##### 错误类型

| 错误 | HTTP 状态码 | 说明 |
|---|---:|---|
| `invalid_request` | 400 | 请求格式、字段、路径或入口声明无效 |
| `execution_failed` | 400 | 导入入口文件或执行入口函数时抛出异常 |
| `invalid_return_value` | 400 | 入口函数返回值不符合协议 |
| `timeout` | 408 | 执行时间超过 `timeout_seconds` |
| `unavailable` | 503 | 暂时没有可用工作进程，或工作进程崩溃 |
| `internal_error` | 500 | 合法请求触发服务端内部错误 |

`invalid_return_value` 包括不支持的 Python 类型、无法导入或非连续的 DLPack 张量、非字符串字典键、循环引用、非有限浮点数以及超过返回值资源限制。返回值满足协议但同步或序列化仍然失败时，服务端返回 `internal_error`。

执行失败响应包含 Python 调用栈：

```json
{
  "status": "error",
  "error": "execution_failed",
  "message": "main() raised RuntimeError: correctness check failed",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "traceback": "...",
  "stdout": "...",
  "stderr": "..."
}
```

调用栈可能泄露源代码和本地路径。面向不可信调用方的部署应关闭调用栈详情或对其进行清理。

### 2.2 `GET /health`

返回服务状态、GPU 工作进程状态和排队请求数量。

#### 2.2.1 请求格式

`GET /health` 不接收请求体、查询参数或其他执行设置：

```http
GET /health HTTP/1.1
Host: server:8000
```

#### 2.2.2 响应

成功响应的内容类型为 `application/json`：

```json
{
  "status": "ok",
  "gpu_count": 2,
  "queue_length": 3,
  "workers": [
    {
      "gpu_id": 0,
      "status": "busy",
      "uptime_seconds": 3600
    },
    {
      "gpu_id": 1,
      "status": "idle",
      "uptime_seconds": 3580
    }
  ]
}
```

工作进程状态取值：

| 状态 | 说明 |
|---|---|
| `idle` | 可以接收请求 |
| `busy` | 正在执行请求 |
| `restarting` | 正在替换超时或失败的运行进程 |
| `unhealthy` | 无法执行请求 |

只要至少一个工作进程能够继续执行请求，服务端就返回 HTTP 200。响应可以同时包含 `restarting` 或 `unhealthy` 工作进程。健康检查只读取服务端状态，每次调用不会额外执行 GPU 内核。

#### 2.2.3 错误

没有任何工作进程能够执行请求时，服务端返回 HTTP 503：

```json
{
  "status": "error",
  "error": "unavailable",
  "message": "no healthy GPU worker is available"
}
```

服务端无法读取内部状态时返回 HTTP 500，错误类型为 `internal_error`。

---

## 3. Python 客户端 SDK

第一版提供同步 Python 客户端，其 `execute()` 方法与同步 `POST /execute` 接口一致。

SDK 是协议的便利封装，不构成安全边界。服务端会重复执行全部验证。


### 3.1 客户端

主要调用方式：

```python
from benchmark_server import Client, Entry


with Client("http://server:8000") as client:
    result = client.execute(
        files={
            "main.py": benchmark_source,
            "submission/kernel.py": kernel_source,
            "data/input.bin": input_bytes,
        },
        language="python",
        entry=Entry(file="main.py", function="main"),
        timeout_seconds=60,
    )
```

公开调用签名：

| API | 约定 |
|---|---|
| `Client(base_url, *, headers=None, connect_timeout_seconds=10)` | 为 `base_url` 创建客户端。可选的 `headers` 随每个请求发送。 |
| `client.execute(files, *, language="python", entry=Entry(), timeout_seconds=None, stdout_limit_bytes=None, stderr_limit_bytes=None) -> ExecutionResult` | 上传文件并同步执行入口函数。 |
| `client.health() -> Health` | 同步读取服务端和工作进程的健康状态。 |
| `client.close() -> None` | 释放客户端的 HTTP 传输资源。 |
| `client.__enter__() -> Client` | 返回已打开的客户端。 |
| `client.__exit__(...) -> None` | 退出上下文时调用 `close()`。 |

`Entry` 是冻结数据类，字段为 `file: str = "main.py"` 和 `function: str = "main"`。

常见调用可以省略全部执行默认参数：

```python
from benchmark_server import Client


with Client("http://server:8000") as client:
    result = client.execute(
        files={"main.py": "def main():\n    return 42\n"},
    )
```

SDK 始终发送必需的 `job`，其中包含显式的 `language`、`entry` 和根据 `files` 生成的文件 manifest。`timeout_seconds`、`stdout_limit_bytes` 或 `stderr_limit_bytes` 为 `None` 时，SDK 省略对应字段，由服务端配置的默认值生效。

`connect_timeout_seconds` 只覆盖连接建立过程。SDK 不会根据执行超时自动生成响应读取期限，因为请求上传时间和排队时间不属于服务端执行超时。

### 3.2 文件输入

公开输入类型为 `FileContent = str | bytes | bytearray | memoryview | pathlib.Path`。

| 值类型 | 上传内容 |
|---|---|
| `str` | 使用 UTF-8 编码后的字符串 |
| `bytes` | 原始字节 |
| `bytearray` | 转换后的原始字节 |
| `memoryview` | 视图数据转换后的原始字节 |
| `pathlib.Path` | 从本地文件读取的内容 |

`files` 参数是从远程相对路径到 `FileContent` 值的映射。映射键会成为 `job.files` 的 key 和请求工作目录中的路径。普通 `str` 值始终表示 UTF-8 文件内容；调用方必须显式使用 `pathlib.Path` 表示本地文件。

SDK 读取或编码每个值，计算完整 SHA-256，将路径和 hash 写入 `job.files`，并把每个唯一内容作为 `blob:<sha256>` part 发送。内容相同的多个路径只产生一个 blob part。所有 blob part 的内容类型都是 `application/octet-stream`。

SDK 在每次请求中发送 manifest 引用的所有唯一 blob，不根据服务端缓存状态省略上传。发送请求前，客户端验证路径安全性，确认 `entry.file` 存在于 `files` 中，要求调用方提供的 `timeout_seconds` 为正数，并要求两个输出长度限制为非负整数。Python 映射中无法同时存在重复键，但 SDK 仍必须规范化全部远程路径，并拒绝规范化后发生的任何冲突。这些客户端检查可以改善错误信息；服务端会重新计算 hash、验证 manifest，并拥有最终决定权。

### 3.3 执行结果

`client.execute()` 返回冻结的 `ExecutionResult` 数据类，字段如下：

| 字段 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `request_id` | `str` | 必需 | 服务端返回的请求标识 |
| `value` | `ReturnValue` | 必需 | 递归解码后的入口函数返回值 |
| `elapsed_ms` | `float` | 必需 | 2.1.2 节定义的服务端执行总时间 |
| `queue_ms` | `float` | 必需 | 2.1.2 节定义的工作进程排队时间 |
| `stdout` | `str` | 必需 | 捕获的标准输出 |
| `stderr` | `str` | 必需 | 捕获的标准错误 |
| `stdout_truncated` | `bool` | `False` | 标准输出是否被截断 |
| `stderr_truncated` | `bool` | `False` | 标准错误是否被截断 |

字段可以直接访问：

```python
print(result.request_id)
print(result.value)
print(result.elapsed_ms, result.queue_ms)
print(result.stdout, result.stderr)
```

`ReturnValue` 按以下规则递归解码：

| 描述节点 | 解码后的 Python 值 |
|---|---|
| `json` | 普通 Python JSON 值：`None`、`bool`、`int`、有限的 `float`、`str`、`list` 和键为字符串的 `dict` |
| `bytes` | `bytes` |
| `tensor` | `tvm_ffi.Tensor` |
| `list` | `list[ReturnValue]` |
| `tuple` | `tuple[ReturnValue, ...]` |
| `dict` | `dict[str, ReturnValue]` |

对于 multipart 响应，客户端通过节点的 `part` 字段查找每个二进制值，并校验其 `size` 和完整 SHA-256 哈希。客户端拒绝重复、缺失和未被引用的二进制部分。客户端还会验证响应体中的 `request_id` 是标准 UUID v4，并且与 `X-Request-ID` 响应头完全一致。

SDK 从不使用 `pickle`。JSON 元数据、multipart 结构、哈希、描述树、二进制部分引用或请求标识不符合协议时，SDK 抛出 `ProtocolError`。

### 3.4 张量

SDK 将 `apache-tvm-ffi` 声明为必需的运行时依赖。TVM FFI 指 TVM 外部函数接口（Foreign Function Interface），它提供 `tvm_ffi.Tensor` 类。DLPack 是用于内存张量交换的标准协议。

无论入口函数返回哪一种受支持的 DLPack 生产者，解码后的张量节点都返回 `tvm_ffi.Tensor`。由于网络传输已经把原始字节复制到主机内存，该张量驻留在中央处理器（CPU）内存中并采用 C 连续的行优先布局。其形状和数据类型与响应元数据完全一致。SDK 拥有的底层存储会在 `tvm_ffi.Tensor` 的整个生命周期中保持有效。

调用方可以通过 DLPack 转换该值：

```python
import numpy as np
import torch
import tvm_ffi


value = result.value
assert isinstance(value, tvm_ffi.Tensor)
np_value = np.from_dlpack(value)
torch_value = torch.from_dlpack(value)
```

NumPy 和 PyTorch 是调用方的可选依赖。仅持有 `tvm_ffi.Tensor` 不要求 SDK 安装这两个包。

HTTP 负载下载完成后，DLPack 转换在本地以零拷贝方式进行。该转换不会让网络传输变成零拷贝。

官方文档：

- [`tvm_ffi.Tensor`](https://tvm.apache.org/ffi/reference/python/generated/tvm_ffi.Tensor.html)
- [`tvm_ffi.from_dlpack`](https://tvm.apache.org/ffi/reference/python/generated/tvm_ffi.from_dlpack.html)

### 3.5 健康检查

健康接口严格使用以下冻结数据类：

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerHealth:
    gpu_id: int
    status: str
    uptime_seconds: float


@dataclass(frozen=True)
class Health:
    status: str
    gpu_count: int
    queue_length: int
    workers: tuple[WorkerHealth, ...]
```

示例：

```python
from benchmark_server import Client


with Client("http://server:8000") as client:
    health = client.health()

for worker in health.workers:
    print(worker.gpu_id, worker.status, worker.uptime_seconds)
```

`client.health()` 同步调用 `GET /health`，并把 HTTP 200 响应解码为 `Health`。HTTP 503 响应会抛出 `BenchmarkServerError`，其中 `code == "unavailable"`。其他结构化健康检查错误使用同样的 `BenchmarkServerError` 映射。

### 3.6 错误与重试

`BenchmarkServerError` 提供以下字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `status_code` | `int` | HTTP 状态码 |
| `code` | `str` | 服务端 `error` 字段中的错误名称 |
| `message` | `str` | 便于人阅读的服务端消息 |
| `request_id` | `str` 或 `None` | 存在请求标识时使用该值 |
| `stdout` | `str` | 捕获的标准输出；缺失时为空字符串 |
| `stderr` | `str` | 捕获的标准错误；缺失时为空字符串 |
| `traceback` | `str` 或 `None` | 存在 Python 调用栈时使用该值 |

失败情况映射到以下异常：

| 失败情况 | 异常 |
|---|---|
| 服务端返回的结构化错误响应 | `BenchmarkServerError` |
| 连接建立、上传、下载、连接中断或其他传输失败 | `TransportError` |
| JSON、multipart 数据、二进制大小或哈希、返回值树或请求标识匹配关系不符合协议 | `ProtocolError` |
| 本地参数无效 | `ValueError` |

SDK 不会自动重试 `POST /execute`。请求到达服务端后如果连接失败，即使客户端没有收到响应，执行也可能已经完成。上层调用方只有在应用语义允许重复执行时才可以重试。

`timeout_seconds` 是服务端执行超时，不包含请求上传和排队。SDK 不会据此生成较短的响应读取超时。

---

## 4. 执行模型

### 4.1 进程与调度

#### 进程模型

服务端包含：

- 一个接收 HTTP 请求并调度任务的前端进程；
- 每张已配置 GPU 对应一个工作槽位；
- 每张 GPU 同一时间最多执行一个请求。

前端进程不初始化 GPU 运行环境。每个工作进程通过 CUDA 设备可见性设置绑定到一张物理 GPU。CUDA 是工作环境使用的 GPU 编程和运行平台。

#### 请求生命周期

每个请求按以下步骤执行：

1. 生成请求 UUID，并建立请求日志上下文。
2. 解析并验证多段请求体。
3. 验证 `job.files` manifest、所有目标路径、入口文件以及 blob 引用集合。
4. 重新计算每个 blob 的 SHA-256，验证 hash，并获取缓存文件引用。
5. 等待一个空闲 GPU 工作进程。
6. 创建新的请求工作目录。
7. 按 `job.files` manifest 在工作目录中放置文件。
8. 设置当前工作目录和 Python 模块搜索路径。
9. 导入入口文件。
10. 获取并验证入口函数。
11. 无参数调用入口函数。
12. 同步并序列化返回值。
13. 捕获标准输出和标准错误。
14. 删除工作目录并释放缓存文件引用。
15. 在响应头和响应元数据中返回请求 UUID。

成功、脚本失败、序列化失败和超时都会执行清理流程。

#### 调度

前端维护先进先出的请求队列。多个 GPU 工作进程同时空闲时，前端按轮转顺序选择下一个工作进程。

每个工作进程串行执行请求，防止多个基准测试请求同时共享一张 GPU，减少性能测量受到的干扰。

`queue_ms` 统计从请求验证完成到分配工作进程之间的时间。

#### 超时与恢复

超时范围包含：

- 导入入口文件；
- 执行文件顶层代码；
- 执行入口函数；
- 同步并序列化返回值。

排队时间不计入执行超时。

超时发生后，服务端执行：

1. 终止正在执行上传脚本的进程；
2. 等待一段较短且可配置的退出宽限时间；
3. 如果进程仍未退出，则强制结束；
4. 在该 GPU 接收下一个请求前替换受影响的运行进程；
5. 删除本次请求工作目录；
6. 返回 `timeout` 错误。

服务端不尝试恢复已经超时的 Python 代码。

### 4.2 文件缓存

文件缓存只用于优化传输和工作目录构建。

缓存键是文件内容的完整 SHA-256 哈希，缓存条目不可修改。两个请求上传完全相同的字节时，可以引用同一缓存文件，同时在各自工作目录中使用不同路径。

缓存只优化服务端内部文件存储和工作目录构建。每个请求仍必须上传 manifest 引用的全部唯一 blob，请求是否有效不依赖当前缓存内容。

文件缓存：

- 不改变入口函数可见的文件路径；
- 不保留脚本对工作目录中文件的修改；
- 不缓存入口函数返回值；
- 不缓存已导入的 Python 模块；
- 不缓存 GPU 内存。

工作目录中的文件必须采用私有副本、写时复制视图或具有同等隔离效果的实现。服务端不能把指向共享缓存的可写硬链接或符号链接直接暴露给脚本。

缓存驱逐只能删除没有活跃请求引用的条目。

### 4.3 隔离

以下状态只属于一个请求：

- 工作目录；
- 上传文件布局；
- 导入的入口模块；
- 捕获的标准输出和标准错误；
- 返回的 Python 对象；
- 只能从本次请求对象访问的 GPU 内存。

服务端不提供会话、跨请求 Python 对象、函数句柄、寄存器或应用层共享状态。

文件内容缓存可以跨请求存活。缓存只保存不可变的上传字节，不保存 Python 模块、GPU 张量或执行结果。

Python 和原生动态链接库可能创建进程级全局状态。实现必须保证请求无法观察到上一个请求导入的入口模块或工作目录。如果长期运行的工作进程无法彻底清理这些状态，在执行下一个请求前必须替换脚本运行进程。

工作目录是请求私有目录。脚本可以读取、修改、删除或创建其中的文件，所有修改只对当前请求可见。请求结束后，服务端删除整个工作目录。

---

## 5. 可观测性

服务端通过部署配置 `log_dir` 指定日志根目录。日志不属于 HTTP API，Python SDK 也不读取日志目录。

### 5.1 日志目录

每次服务进程启动时创建一个独立运行目录：

```text
<log_dir>/
└── runs/
    ├── 20260716T235012.123456Z/
    │   ├── events.jsonl
    │   ├── server.stdout.log
    │   ├── server.stderr.log
    │   └── requests/
    │       └── <request_id>/
    │           ├── stdout.log
    │           └── stderr.log
    └── 20260716T235012.123456Z-1/
        └── ...
```

运行目录名称使用服务启动时的协调世界时（UTC）时间戳，精确到微秒。创建目录时直接调用原子的 `mkdir(..., exist_ok=False)`；如果名称已经存在，则依次尝试 `-1`、`-2` 等数字后缀。生成日志目录不需要文件锁，也不能先检查目录是否存在再创建。

每次重启都创建新目录，不继续写入之前运行的日志文件。多个服务进程同时启动时也会得到不同目录。防止多个服务进程占用同一 GPU 或共享缓存属于资源管理问题，可以独立使用文件锁。

### 5.2 结构化事件

`events.jsonl` 使用 JSON Lines 格式，每行保存一个完整 JSON 对象。所有工作进程把结构化事件发送给前端进程，由前端作为唯一写入者追加文件。每条事件通过一次写操作写入并立即刷新。

定义以下事件：

- `server_started`：服务进程启动；
- `request_started`：请求开始处理；
- `request_finished`：请求成功或失败；
- `worker_restarted`：工作进程被替换；
- `server_error`：服务端自身发生错误；
- `server_stopped`：服务正常关闭。

每条事件包含当前运行目录名称 `server_run`。请求相关事件还必须包含 `request_id`。

`request_finished` 示例：

```json
{
  "timestamp": "2026-07-16T23:50:12.123Z",
  "level": "INFO",
  "event": "request_finished",
  "server_run": "20260716T235012.123456Z",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "http_status": 200,
  "error": null,
  "gpu_id": 0,
  "queue_ms": 18.2,
  "elapsed_ms": 1245.6,
  "stdout_path": "requests/7f61b94e-034a-4e80-b67d-eca52bb952cc/stdout.log",
  "stderr_path": "requests/7f61b94e-034a-4e80-b67d-eca52bb952cc/stderr.log"
}
```

失败时，`error` 使用 2.1.3 节定义的错误类型。

### 5.3 标准输出和标准错误

各文件保存：

- `server.stdout.log`：服务进程、工作进程及其依赖直接写入的标准输出；
- `server.stderr.log`：服务进程、工作进程及其依赖直接写入的标准错误；
- `requests/<request_id>/stdout.log`：入口脚本及其子进程的标准输出；
- `requests/<request_id>/stderr.log`：入口脚本及其子进程的标准错误。

运行入口脚本时，服务端在脚本运行进程中按文件描述符重定向标准输出和标准错误，因此 Python、原生动态链接库和子进程写入的内容都会进入对应请求日志。

日志文件始终保存完整输出，不使用 `stdout_limit_bytes`、`stderr_limit_bytes`，也不包含截断字段。API 响应和 SDK 的 `ExecutionResult` 按请求指定的长度限制返回前缀，并通过 `stdout_truncated`、`stderr_truncated` 表明是否截断。

完整日志可能持续占用磁盘空间。部署方负责为 `log_dir` 分配空间，并通过删除整个旧运行目录实施保存期限；服务端不截断单个日志文件。

### 5.4 重启与异常退出

正常关闭时，服务端在当前 `events.jsonl` 写入 `server_stopped`。服务进程被强制结束时，该事件可能缺失，正在执行的请求也可能只有 `request_started` 而没有 `request_finished`。

异常退出后，已经写入的 `events.jsonl`、请求输出和服务端输出均保留。最后一行 JSON 可能不完整，读取方应忽略无法解析的最后一行。下一次启动创建新的运行目录，不修改旧日志。

服务端启动时清理遗留工作目录和未完成的缓存临时文件，保留完整的内容寻址缓存和所有日志。上一次运行中未完成的同步请求不会恢复，客户端会观察到传输失败并自行决定是否重新提交。

---

## 6. 完整示例

### 6.1 上传的 `main.py`

```python
import json
from pathlib import Path


def main():
    config = json.loads(Path("config.json").read_text())
    input_data = Path("data/input.bin").read_bytes()

    # Build and run the GPU workload here.
    output_size = len(input_data)

    return {
        "correct": True,
        "output_size": output_size,
        "warmup_iterations": config["warmup_iterations"],
        "median_ms": 0.128,
    }
```

### 6.2 请求

```bash
curl -X POST http://server:8000/execute \
  -F 'job={"language":"python","entry":{"file":"main.py","function":"main"},"files":{"main.py":{"blob":"<main_sha256>"},"config.json":{"blob":"<config_sha256>"},"data/input.bin":{"blob":"<input_sha256>"}},"timeout_seconds":60,"stdout_limit_bytes":1048576,"stderr_limit_bytes":1048576};type=application/json' \
  -F 'blob:<main_sha256>=@main.py;type=application/octet-stream' \
  -F 'blob:<config_sha256>=@config.json;type=application/octet-stream' \
  -F 'blob:<input_sha256>=@input.bin;type=application/octet-stream'
```

`<main_sha256>`、`<config_sha256>` 和 `<input_sha256>` 分别替换为对应文件内容的完整小写 SHA-256，并在 `job.files` 和 multipart part 名中使用相同值。

### 6.3 响应

```json
{
  "status": "ok",
  "request_id": "7f61b94e-034a-4e80-b67d-eca52bb952cc",
  "return": {
    "type": "json",
    "value": {
      "correct": true,
      "output_size": 1048576,
      "warmup_iterations": 10,
      "median_ms": 0.128
    }
  },
  "elapsed_ms": 842.7,
  "queue_ms": 0.4,
  "stdout": "",
  "stderr": ""
}
```
